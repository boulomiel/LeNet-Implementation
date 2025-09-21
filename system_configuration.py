import dataclasses


@dataclasses.dataclass
class SystemConfiguration:
    #Describes the common system setting needed for reproducible training
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = True  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)