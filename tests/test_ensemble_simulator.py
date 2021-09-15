from ch_sim import EnsembleSimulator

def test_generate_configs():

    # Ensure we properly distribute tasks between jobs
    test_runs = [None] * 36
    sim = EnsembleSimulator(None)
    configs = sim.generate_job_configs(None, runs=test_runs,
        nodeCount=10, runtime=2, maxJobNodes=10, maxJobRuntime=15)

    assert len(configs) == 6
    assert configs[0]["maxRunTime"] == "14:00:00"
    assert configs[-1]["maxRunTime"] == "02:00:00"

    test_runs = [None] * 29
    configs = sim.generate_job_configs(None, runs=test_runs,
        nodeCount=10, runtime=2, maxJobNodes=20, maxJobRuntime=15)

    assert len(configs) == 3
    assert configs[-1]["maxRunTime"] == "02:00:00"
    assert configs[-1]["nodeCount"] == 10
    assert configs[1]['jobRunInds'] == list(range(14,28))
