import json
import unittest

from petlm.dry_run_motion_gateway import DryRunMotionGateway
from petlm.motion_proposal import MotionProposalBridge, SCHEMA


def proposal(action='move',target=None,speed=None,confirmation=True,confidence=0.9):
    return json.dumps({'schema':SCHEMA,'action':action,'target':target,'speed':speed,
                       'confidence':confidence,'requires_confirmation':confirmation,
                       'reason':'测试建议'},ensure_ascii=False)


class MotionProposalBridgeTests(unittest.TestCase):
    def setUp(self):
        self.gateway=DryRunMotionGateway()
        self.gateway.execute({'type':'heartbeat','command_id':'h1','time':0})
        self.bridge=MotionProposalBridge(self.gateway)

    def test_plain_language_and_schema_injection_are_rejected(self):
        with self.assertRaises(ValueError):
            self.bridge.submit('向前走一米','p1',0.1,True,True)
        injected=json.loads(proposal('move',[1,0],0.1))
        injected['shell']='run'
        with self.assertRaises(ValueError):
            self.bridge.submit(json.dumps(injected),'p2',0.1,True,True)

    def test_move_requires_policy_and_operator_confirmation(self):
        value=proposal('move',[1,0],0.1)
        for policy,operator in ((False,False),(True,False),(False,True)):
            result=self.bridge.submit(value,'unused',0.1,policy,operator)
            self.assertEqual(result['status'],'proposal_blocked')
        self.assertEqual(len(self.gateway.audit),1)

    def test_confirmed_move_reaches_dry_run_only(self):
        result=self.bridge.submit(proposal('move',[1,0],0.1),'p1',0.1,True,True)
        self.assertEqual(result['status'],'submitted_to_dry_run_gateway')
        self.assertTrue(result['dry_run'])
        self.assertFalse(result['actuator_enabled'])
        self.assertEqual(self.gateway.simulator.state.mode,'moving')

    def test_stop_and_estop_are_not_delayed_by_confirmation(self):
        stopped=self.bridge.submit(proposal('stop'),'p1',0.1)
        self.assertEqual(stopped['gateway_outcome']['status'],'dry_run_stopped')
        estopped=self.bridge.submit(proposal('estop'),'p2',-10)
        self.assertTrue(estopped['gateway_outcome']['latched'])

    def test_nonfinite_or_oversized_output_is_rejected(self):
        with self.assertRaises(ValueError):
            self.bridge.submit(proposal('move',[float('nan'),0],0.1),'p1',0.1,True,True)
        with self.assertRaises(ValueError):
            self.bridge.submit('x'*4097,'p2',0.1,True,True)

    def test_clarification_never_reaches_gateway(self):
        result=self.bridge.submit(proposal('clarify'),'p1',0.1)
        self.assertEqual(result['status'],'proposal_not_executed')
        self.assertEqual(len(self.gateway.audit),1)


if __name__=='__main__':
    unittest.main()
