from typing import Literal

Model = Literal['bpr', 'greenshields', 'exponential']
Weight = Literal['length', 'travel_time']
Order = Literal['near', 'far', 'random']

TIME_INTERVAL = 0.1

DEFAULT_BPR_ALPHA = 1.919
DEFAULT_BPR_BETA = 6.9373

def exponential(
        weight: Weight = 'length',
        order: Order = 'near',
        simulation: int = 1000,
        human_speed: float = 1.25,
        v_min: float = 0.03,
        exponential_alpha: float = 2.0,
) -> dict:
    title = f'Exponential {weight.capitalize()} {order.capitalize()} {simulation} {human_speed} {v_min} {exponential_alpha}'
    pattern = {
        'model': 'exponential',
        'weight': weight,
        'order': order,
        'simulation': simulation,
        'human_speed': human_speed,
        'time_interval': TIME_INTERVAL,
        'v_min': v_min,
        'exponential_alpha': exponential_alpha,
        'bpr_alpha': DEFAULT_BPR_ALPHA,
        'bpr_beta': DEFAULT_BPR_BETA,
    }
    return {
        'title': title,
        'pattern': pattern,
    }

def bpr(
        weight: Weight = 'length',
        order: Order = 'near',
        simulation: int = 1000,
        human_speed: float = 1.25,
        bpr_alpha: float = DEFAULT_BPR_ALPHA,
        bpr_beta: float = DEFAULT_BPR_BETA,
) -> dict:
    title = f'BPR {weight.capitalize()} {order.capitalize()} {simulation} {human_speed} {bpr_alpha} {bpr_beta}'
    pattern = {
        'model': 'bpr',
        'weight': weight,
        'order': order,
        'simulation': simulation,
        'human_speed': human_speed,
        'time_interval': TIME_INTERVAL,
        'v_min': 0.03,
        'exponential_alpha': 2.0,
        'bpr_alpha': bpr_alpha,
        'bpr_beta': bpr_beta,
    }
    return {
        'title': title,
        'pattern': pattern,
    }

def greenshields(
        weight: Weight = 'length',
        order: Order = 'near',
        simulation: int = 1000,
        human_speed: float = 1.25,
        v_min: float = 0.03,
) -> dict:
    title = f'Greenshields {weight.capitalize()} {order.capitalize()} {simulation} {human_speed} {v_min}'
    pattern = {
        'model': 'greenshields',
        'weight': weight,
        'order': order,
        'simulation': simulation,
        'human_speed': human_speed,
        'time_interval': TIME_INTERVAL,
        'v_min': v_min,
        'exponential_alpha': 2.0,
        'bpr_alpha': DEFAULT_BPR_ALPHA,
        'bpr_beta': DEFAULT_BPR_BETA,
    }
    return {
        'title': title,
        'pattern': pattern,
    }