"""Structured, persistent, non-actuating motion gateway.

This module deliberately has no serial, GPIO, network, or motor integration.
"""
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import threading

from .motion_simulator import MotionSafetySimulator


COMMAND_FIELDS={
    'heartbeat':{'type','command_id','time'},
    'move':{'type','command_id','time','target','speed'},
    'stop':{'type','command_id','time','reason'},
    'estop':{'type','command_id','time','reason'},
    'reset':{'type','command_id','time','operator_id','operator_authorized'},
    'tick':{'type','command_id','time','dt'},
    'fault':{'type','command_id','time','fault'},
}


class AuditRecoveryError(RuntimeError):
    """Raised when persisted audit evidence cannot be trusted or replayed."""


class DryRunMotionGateway:
    def __init__(self,watchdog_timeout=0.5,max_speed=0.20,max_coordinate=1000.0,
                 max_commands_per_window=100,rate_window=1.0,audit_path=None):
        numeric=(watchdog_timeout,max_coordinate,rate_window)
        if (not all(isinstance(value,(int,float)) and math.isfinite(value) for value in numeric)
                or watchdog_timeout<=0 or max_coordinate<=0 or rate_window<=0):
            raise ValueError('Gateway limits must be positive and finite.')
        if not isinstance(max_commands_per_window,int) or max_commands_per_window<=0:
            raise ValueError('Command rate limit must be a positive integer.')
        self.watchdog_timeout=float(watchdog_timeout)
        self.max_coordinate=float(max_coordinate)
        self.max_commands_per_window=max_commands_per_window
        self.rate_window=float(rate_window)
        self.simulator=MotionSafetySimulator(max_speed=max_speed,max_coordinate=max_coordinate)
        self.actuator_enabled=False
        self.last_heartbeat=None
        self.last_command_time=None
        self.command_ids=set()
        self.audit=[]
        self._last_hash='0'*64
        self._recent_command_times=deque()
        self._lock=threading.RLock()
        self.audit_path=Path(audit_path) if audit_path is not None else None
        if self.audit_path is not None and self.audit_path.exists() and self.audit_path.stat().st_size:
            raise FileExistsError('Audit path already contains data; use recover() to resume safely.')

    @staticmethod
    def _canonical(value):
        return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)

    @classmethod
    def _verify_rows(cls,rows):
        previous='0'*64
        for index,row in enumerate(rows,1):
            if not isinstance(row,dict) or set(row)!={'sequence','previous_hash','command','outcome','actuator_enabled','entry_hash'}:
                return False
            core={key:row[key] for key in ('sequence','previous_hash','command','outcome','actuator_enabled')}
            try:
                digest=hashlib.sha256(cls._canonical(core).encode('utf-8')).hexdigest()
            except (TypeError,ValueError):
                return False
            if (row['sequence']!=index or row['previous_hash']!=previous
                    or row['entry_hash']!=digest or row['actuator_enabled'] is not False):
                return False
            previous=digest
        return True

    def _append_persistent_row(self,row):
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True,exist_ok=True)
        line=self._canonical(row)+'\n'
        with self.audit_path.open('a',encoding='utf-8',newline='\n') as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _record(self,command,outcome):
        core={'sequence':len(self.audit)+1,'previous_hash':self._last_hash,
              'command':command,'outcome':outcome,'actuator_enabled':False}
        digest=hashlib.sha256(self._canonical(core).encode('utf-8')).hexdigest()
        row=dict(core,entry_hash=digest)
        self._append_persistent_row(row)
        self.audit.append(row)
        self._last_hash=digest
        return outcome

    def _check_rate(self,kind,timestamp):
        if kind=='estop':
            return
        boundary=timestamp-self.rate_window
        while self._recent_command_times and self._recent_command_times[0]<=boundary:
            self._recent_command_times.popleft()
        if len(self._recent_command_times)>=self.max_commands_per_window:
            raise RuntimeError('Command rate limit exceeded.')
        self._recent_command_times.append(timestamp)

    def _validate(self,command):
        if not isinstance(command,dict):
            raise TypeError('Only structured command objects are accepted.')
        kind=command.get('type')
        if kind not in COMMAND_FIELDS:
            raise ValueError('Unknown command type.')
        missing=COMMAND_FIELDS[kind]-set(command)
        extra=set(command)-COMMAND_FIELDS[kind]
        if missing or extra:
            raise ValueError(f'Command schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
        command_id=command['command_id']
        if not isinstance(command_id,str) or not command_id.strip() or len(command_id)>128:
            raise ValueError('command_id must contain 1 to 128 characters.')
        duplicate=command_id in self.command_ids
        if duplicate and kind!='estop':
            raise RuntimeError('Duplicate command_id rejected.')
        timestamp=command['time']
        if (not isinstance(timestamp,(int,float)) or isinstance(timestamp,bool)
                or not math.isfinite(timestamp)):
            raise ValueError('Command time must be finite.')
        timestamp=float(timestamp)
        self.command_ids.add(command_id)
        if kind!='estop' and self.last_command_time is not None and timestamp<self.last_command_time:
            raise ValueError('Command time cannot move backwards.')
        self._check_rate(kind,timestamp)
        self.last_command_time=(timestamp if self.last_command_time is None
                                else max(timestamp,self.last_command_time))
        return kind,timestamp,duplicate

    def poll_watchdog(self,now):
        if (not isinstance(now,(int,float)) or not math.isfinite(now)):
            raise ValueError('Watchdog time must be finite.')
        if (self.simulator.state.mode=='moving' and
                (self.last_heartbeat is None or now-self.last_heartbeat>self.watchdog_timeout)):
            self.simulator.emergency_stop('watchdog_timeout')
            return True
        return False

    @staticmethod
    def _safe_copy(command):
        if not isinstance(command,dict):
            return command
        try:
            return json.loads(json.dumps(command,ensure_ascii=False,allow_nan=False))
        except (TypeError,ValueError):
            return {'unserializable_command':repr(command)[:512]}

    def execute(self,command):
        with self._lock:
            safe_copy=self._safe_copy(command)
            if (isinstance(safe_copy,dict)
                    and set(safe_copy)=={'unserializable_command'}):
                exc=ValueError('Command must contain JSON-serializable finite values.')
                self._record(safe_copy,{'status':'rejected','error_type':'ValueError','error':str(exc)})
                raise exc
            try:
                kind,now,duplicate=self._validate(command)
                if kind!='estop' and self.poll_watchdog(now):
                    raise RuntimeError('Watchdog timeout latched emergency stop.')
                if kind=='heartbeat':
                    self.last_heartbeat=now
                    outcome={'status':'heartbeat_accepted'}
                elif kind=='move':
                    if self.last_heartbeat is None or now-self.last_heartbeat>self.watchdog_timeout:
                        self.simulator.emergency_stop('missing_or_stale_heartbeat')
                        raise RuntimeError('Fresh heartbeat required before motion.')
                    target=command['target']
                    if (not isinstance(target,(list,tuple)) or len(target)!=2
                            or not all(isinstance(value,(int,float)) and not isinstance(value,bool)
                                       and math.isfinite(value) for value in target)
                            or any(abs(value)>self.max_coordinate for value in target)):
                        raise ValueError('Target must contain two finite coordinates within the dry-run boundary.')
                    speed=command['speed']
                    if (not isinstance(speed,(int,float)) or isinstance(speed,bool)
                            or not math.isfinite(speed) or speed<=0):
                        raise ValueError('Speed must be positive and finite.')
                    self.simulator.command_move(target,speed)
                    outcome={'status':'dry_run_motion_started','target':list(self.simulator.state.target),
                             'speed':self.simulator.state.speed}
                elif kind=='stop':
                    self.simulator.stop(command['reason'])
                    outcome={'status':'dry_run_stopped'}
                elif kind=='estop':
                    self.simulator.emergency_stop(command['reason'])
                    outcome={'status':'dry_run_emergency_stopped','latched':True,
                             'duplicate_command_id':duplicate}
                elif kind=='reset':
                    if not isinstance(command['operator_id'],str) or not command['operator_id'].strip():
                        raise ValueError('operator_id is required for reset.')
                    self.simulator.reset_estop(command['operator_authorized'] is True)
                    outcome={'status':'dry_run_estop_reset','operator_id':command['operator_id']}
                elif kind=='tick':
                    self.simulator.tick(command['dt'])
                    outcome={'status':'dry_run_tick','mode':self.simulator.state.mode}
                else:
                    self.simulator.inject_fault(command['fault'])
                    outcome={'status':'dry_run_fault_injected','fault':command['fault'],'latched':True}
                return self._record(safe_copy,outcome)
            except Exception as exc:
                self._record(safe_copy,{'status':'rejected','error_type':type(exc).__name__,'error':str(exc)})
                raise

    def verify_audit_chain(self):
        with self._lock:
            return self._verify_rows(self.audit)

    @classmethod
    def recover(cls,audit_path,**settings):
        path=Path(audit_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        except (OSError,json.JSONDecodeError) as exc:
            raise AuditRecoveryError('Persistent audit log is not valid JSONL.') from exc
        if not rows or not cls._verify_rows(rows):
            raise AuditRecoveryError('Persistent audit chain verification failed.')
        replay=cls(**settings)
        for expected in rows:
            if (isinstance(expected['command'],dict)
                    and set(expected['command'])=={'unserializable_command'}):
                replay._record(expected['command'],expected['outcome'])
                continue
            try:
                replay.execute(expected['command'])
            except Exception:
                pass
            actual=replay.audit[-1]
            if actual!=expected:
                raise AuditRecoveryError(f'Audit replay diverged at sequence {expected["sequence"]}.')
        replay.audit_path=path
        return replay

    def snapshot(self):
        with self._lock:
            return {'dry_run':True,'actuator_enabled':False,'watchdog_timeout':self.watchdog_timeout,
                    'max_coordinate':self.max_coordinate,
                    'max_commands_per_window':self.max_commands_per_window,
                    'rate_window':self.rate_window,'last_heartbeat':self.last_heartbeat,
                    'last_command_time':self.last_command_time,'audit_entries':len(self.audit),
                    'audit_head':self._last_hash,'audit_chain_valid':self.verify_audit_chain(),
                    'persistent_audit':str(self.audit_path) if self.audit_path else None,
                    'simulator':self.simulator.snapshot()}
