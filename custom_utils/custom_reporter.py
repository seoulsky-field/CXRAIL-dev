from typing import Any, Callable, Dict, List, Optional, Union
from ray.tune.experiment.trial import DEBUG_PRINT_INTERVAL, Trial, _Location
from ray.tune import CLIReporter


class TrialTerminationReporter(CLIReporter):
    def __init__(self, metric_columns, parameter_columns):
        self.num_terminated = 0
        super(TrialTerminationReporter, self).__init__(
            metric_columns=metric_columns,
            parameter_columns=parameter_columns,
        )

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])

        return self.num_terminated > old_num_terminated

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        print(self._progress_str(trials, done, *sys_info))
