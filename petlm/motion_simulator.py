"""Non-actuating motion safety simulator with latched emergency stop."""
from dataclasses import asdict, dataclass, field
import math


FAULTS=('localization_lost','sensor_offline','sensor_conflict','person_in_path',
        'obstacle_in_path','sensor_stale')


@dataclass
class SensorState:
    localization_ready: bool = True
    obstacle_sensor_ready: bool = True
    readings_consistent: bool = True
    person_in_path: bool = False
    obstacle_in_path: bool = False
    fresh: bool = True

    def healthy(self):
        return (self.localization_ready and self.obstacle_sensor_ready and
                self.readings_consistent and not self.person_in_path and
                not self.obstacle_in_path and self.fresh)


@dataclass
class MotionState:
    x: float = 0.0
    y: float = 0.0
    speed: float = 0.0
    target: tuple | None = None
    mode: str = 'stopped'
    estop_latched: bool = False
    stop_reason: str = 'startup'
    control_cycles: int = 0
    events: list = field(default_factory=list)


class MotionSafetySimulator:
    def __init__(self,max_speed=0.20,arrival_tolerance=0.02,max_coordinate=1000.0):
        if (not all(isinstance(value,(int,float)) and math.isfinite(value)
                    for value in (max_speed,arrival_tolerance,max_coordinate))
                or max_speed<=0 or arrival_tolerance<=0 or max_coordinate<=0):
            raise ValueError('Speed and tolerance must be positive.')
        self.max_speed=float(max_speed)
        self.arrival_tolerance=float(arrival_tolerance)
        self.max_coordinate=float(max_coordinate)
        self.sensors=SensorState()
        self.state=MotionState()

    def _event(self,name,**details):
        self.state.events.append({'cycle':self.state.control_cycles,'event':name,**details})

    def emergency_stop(self,reason):
        self.state.speed=0.0
        self.state.mode='emergency_stopped'
        self.state.estop_latched=True
        self.state.stop_reason=reason
        self._event('emergency_stop',reason=reason)

    def command_move(self,target,speed):
        if (not isinstance(target,(tuple,list)) or len(target)!=2 or
                not all(isinstance(value,(int,float)) and math.isfinite(value) for value in target)):
            raise ValueError('Target must contain two finite coordinates.')
        if any(abs(value)>self.max_coordinate for value in target):
            raise ValueError('Target coordinate exceeds the configured dry-run boundary.')
        if not isinstance(speed,(int,float)) or not math.isfinite(speed) or speed<=0:
            raise ValueError('Commanded speed must be a positive finite number.')
        if self.state.estop_latched:
            raise RuntimeError('Emergency stop is latched; operator reset is required.')
        if not self.sensors.healthy():
            self.emergency_stop('preflight_sensor_check_failed')
            raise RuntimeError('Sensors are not safe for motion.')
        self.state.target=(float(target[0]),float(target[1]))
        self.state.speed=min(float(speed),self.max_speed)
        self.state.mode='moving'
        self.state.stop_reason=''
        self._event('motion_started',target=self.state.target,speed=self.state.speed)

    def stop(self,reason='operator_stop'):
        self.state.speed=0.0
        self.state.mode='stopped'
        self.state.stop_reason=reason
        self._event('controlled_stop',reason=reason)

    def inject_fault(self,fault):
        if fault not in FAULTS:
            raise ValueError(f'Unknown fault: {fault}')
        if fault=='localization_lost': self.sensors.localization_ready=False
        elif fault=='sensor_offline': self.sensors.obstacle_sensor_ready=False
        elif fault=='sensor_conflict': self.sensors.readings_consistent=False
        elif fault=='person_in_path': self.sensors.person_in_path=True
        elif fault=='obstacle_in_path': self.sensors.obstacle_in_path=True
        elif fault=='sensor_stale': self.sensors.fresh=False
        self._event('fault_injected',fault=fault)
        self.emergency_stop(fault)

    def restore_sensors(self):
        self.sensors=SensorState()
        self._event('sensors_restored')

    def reset_estop(self,operator_authorized=False):
        if not operator_authorized:
            raise PermissionError('An on-site operator must authorize emergency-stop reset.')
        if not self.sensors.healthy():
            raise RuntimeError('Cannot reset emergency stop while sensors are unhealthy.')
        self.state.estop_latched=False
        self.state.mode='stopped'
        self.state.speed=0.0
        self.state.stop_reason='operator_reset'
        self._event('emergency_stop_reset')

    def tick(self,dt):
        if not isinstance(dt,(int,float)) or dt<=0 or not math.isfinite(dt):
            raise ValueError('Control interval must be a positive finite number.')
        self.state.control_cycles+=1
        if self.state.mode!='moving':
            return
        if not self.sensors.healthy():
            self.emergency_stop('runtime_sensor_check_failed')
            return
        tx,ty=self.state.target
        dx,dy=tx-self.state.x,ty-self.state.y
        distance=math.hypot(dx,dy)
        if distance<=self.arrival_tolerance:
            self.state.x,self.state.y=tx,ty
            self.stop('target_reached')
            return
        step=min(distance,self.state.speed*dt)
        self.state.x+=dx/distance*step
        self.state.y+=dy/distance*step
        if step>=distance-self.arrival_tolerance:
            self.state.x,self.state.y=tx,ty
            self.stop('target_reached')

    def snapshot(self):
        return {'motion':asdict(self.state),'sensors':asdict(self.sensors),
                'max_speed':self.max_speed,'arrival_tolerance':self.arrival_tolerance,
                'max_coordinate':self.max_coordinate}
