import unittest

from petlm.motion_simulator import FAULTS, MotionSafetySimulator


class MotionSafetySimulatorTests(unittest.TestCase):
    def test_normal_motion_and_speed_limit(self):
        sim=MotionSafetySimulator(max_speed=0.2)
        sim.command_move((1,0),1.0)
        self.assertEqual(sim.state.speed,0.2)
        sim.tick(1.0)
        self.assertAlmostEqual(sim.state.x,0.2)
        self.assertEqual(sim.state.mode,'moving')

    def test_manual_estop_is_immediate_and_latched(self):
        sim=MotionSafetySimulator()
        sim.command_move((1,0),0.1)
        sim.emergency_stop('operator_button')
        self.assertEqual(sim.state.speed,0)
        self.assertTrue(sim.state.estop_latched)
        with self.assertRaises(RuntimeError): sim.command_move((1,0),0.1)

    def test_every_fault_stops_and_latches(self):
        for fault in FAULTS:
            with self.subTest(fault=fault):
                sim=MotionSafetySimulator()
                sim.command_move((1,0),0.1)
                sim.inject_fault(fault)
                self.assertEqual(sim.state.speed,0)
                self.assertTrue(sim.state.estop_latched)
                self.assertEqual(sim.state.stop_reason,fault)

    def test_reset_requires_operator_and_healthy_sensors(self):
        sim=MotionSafetySimulator()
        sim.command_move((1,0),0.1)
        sim.inject_fault('sensor_offline')
        with self.assertRaises(PermissionError): sim.reset_estop(False)
        with self.assertRaises(RuntimeError): sim.reset_estop(True)
        sim.restore_sensors()
        sim.reset_estop(True)
        self.assertFalse(sim.state.estop_latched)
        self.assertEqual(sim.state.mode,'stopped')

    def test_preflight_failure_blocks_motion(self):
        sim=MotionSafetySimulator()
        sim.sensors.localization_ready=False
        with self.assertRaises(RuntimeError): sim.command_move((1,0),0.1)
        self.assertTrue(sim.state.estop_latched)

    def test_invalid_target_and_control_interval_are_rejected(self):
        sim=MotionSafetySimulator()
        with self.assertRaises(ValueError): sim.command_move(('x',0),0.1)
        with self.assertRaises(ValueError): sim.tick(0)


if __name__=='__main__':
    unittest.main()
