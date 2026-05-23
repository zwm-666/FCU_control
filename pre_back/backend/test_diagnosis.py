import tempfile
import unittest

import diagnosis


class OnlineDiagnosisToggleTest(unittest.TestCase):
    def test_predict_uses_rule_based_diagnosis_when_model_prediction_disabled(self):
        with tempfile.TemporaryDirectory() as model_dir:
            diag = diagnosis.OnlineDiagnosis(model_dir=model_dir, enable_model_prediction=False)

        machine_state = {
            "power": {
                "stackVoltage": 48.0,
                "stackCurrent": 20.0,
                "stackPower": 1.5,
            },
            "sensors": {
                "stackTemp": 85.0,
                "ambientTemp": 25.0,
                "h2Concentration": 0.1,
                "airInletTemp": 30.4,
                "airFlow": 6.18,
            },
            "io": {
                "faultCode": 0,
            },
        }

        result = diag.predict(machine_state)
        status = diag.get_status()

        self.assertFalse(diag.is_trained)
        self.assertIsNone(diag.model)
        self.assertEqual(result["label"], "thermal_issue")
        self.assertEqual(result["label_cn"], "热管理异常")
        self.assertFalse(status["model_prediction_enabled"])


if __name__ == "__main__":
    unittest.main()
