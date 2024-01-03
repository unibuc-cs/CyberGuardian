#https://realpython.com/simpy-simulating-with-python/

import simpy
import random
import statistics

wait_times = []


class IoTDevice(object):
    def __init__(self, IP: str, Port: int) -> None:
        self.IP = IP
        self.Port = Port


# The IoT hub has 3 types of processes, each with different hardware process:
# a) a logger
# b) a metrics storer (e.g., flower temperature at every minute, etc)
# c) a rules processors - reads the metrics and take actions according to some predefined rules
class IoTHub(object):
    def __init__(self, env, num_loggers: int, num_store_metrics: int, num_rules_processors: int):
        self.env = env
        self.loggers = simpy.Resource(env, num_loggers)
        self.metrics = simpy.Resource(env, num_store_metrics)
        self.rules = simpy.Resource(env, num_rules_processors)

    def log_state_action(self, entity: IoTDevice):
        yield self.env.timeout(random.uniform(0.01, 0.1))

    def update_metrics_in_database(self, entity: IoTDevice):
        yield self.env.timeout(random.uniform(0.01, 0.2))

    def sell_food(self, moviegoer):
        yield self.env.timeout(random.randint(1, 5))


def go_to_movies(env, moviegoer, theater):
    # Moviegoer arrives at the theater
    arrival_time = env.now

    with theater.cashier.request() as request:
        yield request
        yield env.process(theater.purchase_ticket(moviegoer))

    with theater.usher.request() as request:
        yield request
        yield env.process(theater.check_ticket(moviegoer))

    if random.choice([True, False]):
        with theater.server.request() as request:
            yield request
            yield env.process(theater.sell_food(moviegoer))

    # Moviegoer heads into the theater
    wait_times.append(env.now - arrival_time)


def run_theater(env, num_cashiers, num_servers, num_ushers):
    theater = Theater(env, num_cashiers, num_servers, num_ushers)

    for moviegoer in range(3):
        env.process(go_to_movies(env, moviegoer, theater))

    while True:
        yield env.timeout(0.20)  # Wait a bit before generating a new person

        moviegoer += 1
        env.process(go_to_movies(env, moviegoer, theater))


def get_average_wait_time(wait_times):
    average_wait = statistics.mean(wait_times)
    # Pretty print the results
    minutes, frac_minutes = divmod(average_wait, 1)
    seconds = frac_minutes * 60
    return round(minutes), round(seconds)


def get_user_input():
    num_cashiers = input("Input # of cashiers working: ")
    num_servers = input("Input # of servers working: ")
    num_ushers = input("Input # of ushers working: ")
    params = [num_cashiers, num_servers, num_ushers]
    if all(str(i).isdigit() for i in params):  # Check input is valid
        params = [int(x) for x in params]
    else:
        print(
            "Could not parse input. Simulation will use default values:",
            "\n1 cashier, 1 server, 1 usher.",
        )
        params = [1, 1, 1]
    return params


def main():
    # Setup
    random.seed(42)
    num_cashiers, num_servers, num_ushers = get_user_input()

    # Run the simulation
    env = simpy.Environment()
    env.process(run_theater(env, num_cashiers, num_servers, num_ushers))
    env.run(until=90)

    # View the results
    mins, secs = get_average_wait_time(wait_times)
    print(
        "Running simulation...",
        f"\nThe average wait time is {mins} minutes and {secs} seconds.",
    )


if __name__ == "__main__":
    main()
