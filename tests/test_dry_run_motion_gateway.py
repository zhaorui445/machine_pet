import unittest
from pathlib import Path
import tempfile
import threading

from petlm.dry_run_motion_gateway import AuditRecoveryError, DryRunMotionGateway


def command(kind,identifier,now,**values):
    return {'type':kind,'command_id':identifier,'time':now,**values}


class DryRunMotionGatewayTests(unittest.TestCase):
    def test_rejects_natural_language_and_unknown_fields(self):
        gateway=DryRunMotionGateway()
        with self.assertRaises(TypeError): gateway.execute('向前走')
        with self.assertRaises(ValueError):
            gateway.execute(dict(command('heartbeat','h1',0),instruction='move'))
        self.assertFalse(gateway.actuator_enabled)

    def test_fresh_heartbeat_required(self):
        gateway=DryRunMotionGateway(watchdog_timeout=0.5)
        with self.assertRaises(RuntimeError):
            gateway.execute(command('move','m1',1,target=[1,0],speed=0.1))
        self.assertTrue(gateway.simulator.state.estop_latched)

    def test_watchdog_timeout_estops_moving_system(self):
        gateway=DryRunMotionGateway(watchdog_timeout=0.5)
        gateway.execute(command('heartbeat','h1',0))
        gateway.execute(command('move','m1',0.1,target=[1,0],speed=0.1))
        with self.assertRaises(RuntimeError):
            gateway.execute(command('tick','t1',0.7,dt=0.1))
        self.assertEqual(gateway.simulator.state.stop_reason,'watchdog_timeout')
        self.assertEqual(gateway.simulator.state.speed,0)

    def test_estop_has_priority_and_reset_requires_operator(self):
        gateway=DryRunMotionGateway()
        gateway.execute(command('estop','e1',0,reason='operator_button'))
        with self.assertRaises(PermissionError):
            gateway.execute(command('reset','r1',0.1,operator_id='operator-1',operator_authorized=False))
        gateway.execute(command('reset','r2',0.2,operator_id='operator-1',operator_authorized=True))
        self.assertFalse(gateway.simulator.state.estop_latched)

    def test_duplicate_estop_is_idempotent_and_audited(self):
        gateway=DryRunMotionGateway()
        gateway.execute(command('estop','e1',0,reason='operator_button'))
        outcome=gateway.execute(command('estop','e1',0.1,reason='operator_button_repeat'))
        self.assertTrue(outcome['latched'])
        self.assertTrue(outcome['duplicate_command_id'])
        self.assertTrue(gateway.simulator.state.estop_latched)
        self.assertTrue(gateway.verify_audit_chain())

    def test_fault_latches_and_duplicate_command_is_rejected(self):
        gateway=DryRunMotionGateway()
        gateway.execute(command('fault','f1',0,fault='sensor_conflict'))
        with self.assertRaises(RuntimeError):
            gateway.execute(command('fault','f1',0.1,fault='sensor_conflict'))
        self.assertTrue(gateway.simulator.state.estop_latched)

    def test_audit_chain_detects_tampering(self):
        gateway=DryRunMotionGateway()
        gateway.execute(command('heartbeat','h1',0))
        gateway.execute(command('estop','e1',0.1,reason='test'))
        self.assertTrue(gateway.verify_audit_chain())
        gateway.audit[0]['outcome']['status']='tampered'
        self.assertFalse(gateway.verify_audit_chain())

    def test_persistent_audit_recovers_state_and_rejects_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audit.jsonl'
            gateway=DryRunMotionGateway(audit_path=path)
            gateway.execute(command('heartbeat','h1',0))
            gateway.execute(command('move','m1',0.1,target=[1,0],speed=0.1))
            gateway.execute(command('estop','e1',0.2,reason='test'))
            recovered=DryRunMotionGateway.recover(path)
            self.assertTrue(recovered.verify_audit_chain())
            self.assertTrue(recovered.simulator.state.estop_latched)
            self.assertEqual(recovered.snapshot()['audit_head'],gateway.snapshot()['audit_head'])
            with self.assertRaises(RuntimeError):
                recovered.execute(command('heartbeat','h1',0.3))

    def test_recovery_rejects_tampered_persistent_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audit.jsonl'
            gateway=DryRunMotionGateway(audit_path=path)
            gateway.execute(command('heartbeat','h1',0))
            text=path.read_text(encoding='utf-8').replace('heartbeat_accepted','tampered')
            path.write_text(text,encoding='utf-8')
            with self.assertRaises(AuditRecoveryError):
                DryRunMotionGateway.recover(path)

    def test_out_of_order_and_nonfinite_commands_are_rejected(self):
        gateway=DryRunMotionGateway(max_coordinate=10)
        gateway.execute(command('heartbeat','h1',2))
        with self.assertRaises(ValueError):
            gateway.execute(command('tick','t1',1,dt=0.1))
        for identifier,target,speed in (
                ('huge',[11,0],0.1),('nan-target',[float('nan'),0],0.1),
                ('nan-speed',[1,0],float('nan'))):
            with self.assertRaises(ValueError):
                gateway.execute(command('move',identifier,2,target=target,speed=speed))
        self.assertFalse(gateway.actuator_enabled)

    def test_rate_limit_never_blocks_estop(self):
        gateway=DryRunMotionGateway(max_commands_per_window=2,rate_window=1)
        gateway.execute(command('heartbeat','h1',0))
        gateway.execute(command('heartbeat','h2',0.1))
        with self.assertRaises(RuntimeError):
            gateway.execute(command('heartbeat','h3',0.2))
        outcome=gateway.execute(command('estop','e1',-100,reason='priority'))
        self.assertTrue(outcome['latched'])
        self.assertTrue(gateway.simulator.state.estop_latched)

    def test_watchdog_boundary(self):
        gateway=DryRunMotionGateway(watchdog_timeout=0.5)
        gateway.execute(command('heartbeat','h1',0))
        gateway.execute(command('move','m1',0.1,target=[1,0],speed=0.1))
        gateway.execute(command('tick','boundary',0.5,dt=0.01))
        self.assertFalse(gateway.simulator.state.estop_latched)
        with self.assertRaises(RuntimeError):
            gateway.execute(command('tick','late',0.500001,dt=0.01))
        self.assertEqual(gateway.simulator.state.stop_reason,'watchdog_timeout')

    def test_concurrent_estop_always_leaves_motion_latched(self):
        gateway=DryRunMotionGateway()
        gateway.execute(command('heartbeat','h1',0))
        barrier=threading.Barrier(3)
        errors=[]
        def run(value):
            barrier.wait()
            try:
                gateway.execute(value)
            except RuntimeError:
                errors.append(value['type'])
        moving=threading.Thread(target=run,args=(command('move','m1',0.1,target=[1,0],speed=0.1),))
        stopping=threading.Thread(target=run,args=(command('estop','e1',0.1,reason='concurrent'),))
        moving.start(); stopping.start(); barrier.wait(); moving.join(); stopping.join()
        self.assertTrue(gateway.simulator.state.estop_latched)
        self.assertEqual(gateway.simulator.state.speed,0)
        self.assertTrue(gateway.verify_audit_chain())


if __name__=='__main__':
    unittest.main()
