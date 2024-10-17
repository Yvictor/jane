
import jane.kaggle_evaluation.core.templates

import jane.kaggle_evaluation.jane_street_gateway


class JSInferenceServer(jane.kaggle_evaluation.core.templates.InferenceServer):
    def _get_gateway_for_test(self, data_paths=None):
        return jane.kaggle_evaluation.jane_street_gateway.JSGateway(data_paths)
