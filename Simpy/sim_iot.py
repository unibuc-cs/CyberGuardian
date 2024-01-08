# https://realpython.com/simpy-simulating-with-python/
import time

import pandas
import simpy
import random
import statistics
import ipaddress
import random
from typing import Dict, List, Tuple, Union
from ipaddress import IPv4Network
import random
import sys
import math
import pandas as pd
from enum import IntEnum
from datetime import datetime
import string

################ GENERATOR FUNCTIONS #################
MAX_IPV4 = ipaddress.IPv4Address._ALL_ONES  # 2 ** 32 - 1
MAX_IPV6 = ipaddress.IPv6Address._ALL_ONES  # 2 ** 128 - 1

# Timeouts in seconds
METRICS_UPDATE_TIMEOUT = 200
METRICS_GET_DATA_TIMEOUT = 150

NHOUSES = 10

# If used, the device hacking will try to crawl the database with random searches using GET
# Too many will basically block the server eventually
USE_DEVICE_HACKING = False
START_TIME_HACKING = (60 * 60 * 10)  # At 10:00 AM
PERCENT_OF_HACKED_DEVICES = 0.5  # Approximately 20% of devices
NUM_DEVICES_TO_HACK = int((NHOUSES * 5) * PERCENT_OF_HACKED_DEVICES)

gen_random_word = lambda n: ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(n))


class RequestType(IntEnum):
    REQ_PUT = 1,
    REQ_GET = 2,
    REQ_DELETE = 3,
    REQ_UPDATE = 4


DATASET_LOGS = pd.DataFrame({
    "id": pd.Series(dtype=str),
    "start_t": pd.Series(dtype=int),
    "end_t": pd.Series(dtype=int),
    "ip": pd.Series(dtype='str'),
    "long": pd.Series(dtype='str'),
    "lat": pd.Series(dtype='str'),
    "request_type": pd.Series(dtype=int),
    "request_params": str,
    "response": pd.Series(dtype=int)})


def random_ipv4():
    return ipaddress.IPv4Address._string_from_ip_int(
        random.randint(0, MAX_IPV4)
    )


class Loc:
    def __init__(self, long, lat):
        self.long = long
        self.lat = lat


def generate_random_LocData(lat, lon, num_rows) -> List[Loc]:
    res = []
    for _ in range(num_rows):
        hex1 = '%012x' % random.randrange(16 ** 12)  # 12 char random string
        flt = float(random.randint(0, 100))
        dec_lat = random.random() / 100
        dec_lon = random.random() / 100
        res.append(Loc(lon + dec_lon, lat + dec_lat))
    return res


def generate_random_ipAddresses(num: int) -> List[str]:
    return [random_ipv4() for _ in range(num)]


allDevices = []
allDevicesByHouse = {}



#############################

class IoTDevice(object):
    def __init__(self, env: simpy.Environment, IP: str, Port: int, ParentHub, id: str, loc_lat, loc_long) -> None:
        self.env = env
        self.IP = IP
        self.Port = Port
        self.hub = ParentHub
        self.id = id
        self.loc_lat = loc_lat
        self.loc_long = loc_long

        self.duration_between_metrics_update = None
        self.last_metrics_update_time = 0

        self.duration_between_data_retrieve = None
        self.last_data_retrieve_time = 0
        self.isHacked = False

    def start(self):
        self.action_get_data = self.env.process(self.run_get_data())
        self.action_send_update = self.env.process(self.run_send_update())

    def run_get_data(self):
        while True:
            # Getting data
            #################################################################################
            cycle_start_time = self.env.now

            next_data_retrieval = self.last_data_retrieve_time + self.duration_between_data_retrieve
            if next_data_retrieval > cycle_start_time:
                yield self.env.timeout(next_data_retrieval - cycle_start_time)
            cycle_start_time = self.env.now
            self.last_data_retrieve_time= cycle_start_time

            # print(f"device: {self.id} starting to update metrics at {self.env.now}")

            # print(f"device: {self.id} starting to log at {cycle_start_time}")
            with self.hub.dataretrieval.request() as request:
                yield request | self.env.timeout(METRICS_GET_DATA_TIMEOUT)

                res = True
                if request.triggered:
                    yield self.env.process(self.hub.get_data(self))
                else:
                    res = False

            DATASET_LOGS.loc[len(DATASET_LOGS)] = {
                "id": self.id,
                "start_t": cycle_start_time,
                "end_t": self.env.now,
                "ip": self.IP,
                "long": self.loc_long,
                "lat": self.loc_lat,
                "request_type": RequestType.REQ_GET,
                "request_params": f"data_{cycle_start_time / 100.0:.2f}" if not self.isHacked
                else gen_random_word(random.randint(5, 10)),
                "response": random.choice([503, 000]) if res is False else 200}

    def run_send_update(self):
        while True:
            # Metrics update
            #################################################################################
            cycle_start_time = self.env.now

            next_metrics_update = self.last_metrics_update_time + self.duration_between_metrics_update
            if next_metrics_update > cycle_start_time:
                yield self.env.timeout(next_metrics_update - cycle_start_time)
            cycle_start_time = self.env.now
            self.last_metrics_update_time = cycle_start_time

            # print(f"device: {self.id} starting to update metrics at {self.env.now}")

            # Then try to update the metrics
            with self.hub.metrics.request() as request:
                yield request | self.env.timeout(METRICS_UPDATE_TIMEOUT)

                res = True
                if request.triggered:
                    yield self.env.process(self.hub.update_metrics_in_database(self))
                else:
                    res = False
            # print(f"device: {self.id} ended to update metrics at {self.env.now}")

            DATASET_LOGS.loc[len(DATASET_LOGS)] = {
                "id": self.id,
                "start_t": cycle_start_time,
                "end_t": self.env.now,
                "ip": self.IP,
                "long": self.loc_long,
                "lat": self.loc_lat,
                "request_type": RequestType.REQ_PUT,
                "request_params": f"metric_update{cycle_start_time / 100.0:.2f}",
                "response": random.choice([503, 000]) if res is False else 200}

    def startHack(self):
        self.isHacked = True

        # Dummy, reduce the time between data retrievals
        newTimeBetweenDataRetrieve = self.duration_between_data_retrieve * (1.0 / random.randint(50, 100))
        self.duration_between_data_retrieve = newTimeBetweenDataRetrieve


def data_factor_ifhacked(device: IoTDevice) ->int :
    return random.randint(10,50) if device.isHacked else 1

def rate_factor_ifhacked(device: IoTDevice) ->int :
    return random.randint(50,100) if device.isHacked else 1


class IoTDeviceSmartTV(IoTDevice):
    def __init__(self, env: simpy.Environment, IP: str, Port: int, ParentHub, id: str, loc_lat, loc_long):
        super().__init__(env, IP, Port, ParentHub, id, loc_lat, loc_long)
        self.duration_between_metrics_update = 2000
        self.duration_between_data_retrieve = 2000


class IoTDeviceSmartKettle(IoTDevice):
    def __init__(self, env: simpy.Environment, IP: str, Port: int, ParentHub, id: str, loc_lat, loc_long):
        super().__init__(env, IP, Port, ParentHub, id, loc_lat, loc_long)
        self.duration_between_metrics_update = 6000
        self.duration_between_data_retrieve =  6000


class IoTDeviceSmartBrush(IoTDevice):
    def __init__(self, env: simpy.Environment, IP: str, Port: int, ParentHub, id: str, loc_lat, loc_long):
        super().__init__(env, IP, Port, ParentHub, id, loc_lat, loc_long)
        self.duration_between_metrics_update = 8000
        self.duration_between_data_retrieve = 8000


class IoTDeviceSmartFlower(IoTDevice):
    def __init__(self, env: simpy.Environment, IP: str, Port: int, ParentHub, id: str, loc_lat, loc_long):
        super().__init__(env, IP, Port, ParentHub, id, loc_lat, loc_long)
        self.duration_between_metrics_update = 4000
        self.duration_between_data_retrieve = 4000


class IoTDeviceSmartWindow(IoTDevice):
    def __init__(self, env: simpy.Environment, IP: str, Port: int, ParentHub, id: str, loc_lat, loc_long):
        super().__init__(env, IP, Port, ParentHub, id, loc_lat, loc_long)
        self.duration_between_metrics_update = 2000
        self.duration_between_data_retrieve = 2800

# The IoT hub has 3 types of processes, each with different hardware process:
# a) a logger
# b) a metrics storer (e.g., flower temperature at every minute, etc)
# c) a rules processors - reads the metrics and take actions according to some predefined rules
class IoTHub(object):
    time_between_updaterules = 1000.0  # 3 seconds between updates


    # The parameters represent the number of parallel processes that are able to run for different operations
    def __init__(self, env, num_loggers: int, num_store_metrics: int, num_rules_processors: int, ip: str, long: str,
                 lat: str):
        self.env = env
        self.dataretrieval = simpy.Resource(env, num_loggers)
        self.metrics = simpy.Resource(env, num_store_metrics)
        self.rules = simpy.Resource(env, num_rules_processors)

        self.last_time_rules_checked = 0.0
        self.action = None
        self.IP = ip
        self.loc_long = long
        self.loc_lat = lat

        # Occupancy of the hub resources
        self.resourcesOccupancy = pandas.DataFrame(
            {'time': pd.Series(dtype=int),
             'dataretrieval_count': pd.Series(dtype=int),
             'dataretrieval_waiting': pd.Series(dtype=int),
             'dataretrieval_occupancy': pd.Series(dtype=float),
             'datauptader_count': pd.Series(dtype=int),
             'dataupdater_waiting': pd.Series(dtype=int),
             'dataupdater_occupancy': pd.Series(dtype=float)
             })


        MONITOR_RATE = 1000
        self.env.process(self.monitor_resources(MONITOR_RATE))

    def monitor_resources(self, SAMPLING_RATE):
        while True:
            item = [self.env.now,
                    self.dataretrieval.count,
                    len(self.dataretrieval.queue),
                    -1,
                    self.metrics.count,
                    len(self.metrics.queue),
                    -1]
            item[3] = min(1.0, item[2]/item[1]) if item[1] > 0 else 0.0
            item[6] = min(1.0, item[5]/item[4]) if item[4] > 0 else 0.0

            self.resourcesOccupancy.loc[len(self.resourcesOccupancy)] = item
            yield self.env.timeout(SAMPLING_RATE)


    def start(self):
        self.action = self.env.process(self.run())

    def get_data(self, entity: IoTDevice):
        # print(f"Logging action for entity: {entity.id}")
        yield self.env.timeout(random.uniform(30, 60) * data_factor_ifhacked(entity))

    def update_metrics_in_database(self, entity: IoTDevice):
        # print(f"Updating metrics for entity: {entity.id}")
        yield self.env.timeout(random.uniform(30, 60)* data_factor_ifhacked(entity))

    def check_update_rules(self) -> bool:
        # print(f"Hub updating rules starting at {self.env.now}...")
        self.last_time_rules_checked = self.env.now
        yield self.env.timeout(random.uniform(1000, 2000))
        return True
        # TODO: this should action back on devices !

    def run(self):
        def getTimeToNextRuleCheck() -> float:
            return (IoTHub.time_between_updaterules + self.last_time_rules_checked) - self.env.now

        while True:
            time_up_to_next_rule_check = getTimeToNextRuleCheck()

            if time_up_to_next_rule_check > 0:
                yield self.env.timeout(time_up_to_next_rule_check)

            start_t = self.env.now

            assert getTimeToNextRuleCheck() <= 0, "it should have slept.."
            with self.rules.request() as request:
                yield request
                yield self.env.process(self.check_update_rules())

            # print(f"Finished rules check: at {self.env.now}")
            DATASET_LOGS.loc[len(DATASET_LOGS)] = {
                "id": "IoTHUB",
                "start_t": start_t,
                "end_t": self.env.now,
                "ip": self.IP,
                "long": self.loc_long,
                "lat": self.loc_lat,
                "request_type": RequestType.REQ_PUT,
                "request_params": "statusupdatelog",
                "response": 200}


class IoTHacker(object):
    def __init__(self, env: simpy.Environment, timeToStartHackingAt: int, numDevicesToHack: int) -> None:
        self.env = env
        self.timeToStartHackingAt = timeToStartHackingAt
        self.numDevicesToHack = numDevicesToHack

        self.action = self.env.process(self.run())

    def run(self):
        # Wait a bit...
        yield self.env.timeout(self.timeToStartHackingAt)

        # Then start
        devicesToHack = random.sample(allDevices, k=self.numDevicesToHack)
        for device in devicesToHack:
            device.startHack()


def generateRandomDeployment(env: simpy.Environment,
                             globalHub: IoTHub,
                             nhouses: int,
                             center_lat: float,
                             center_long: float):
    locations = generate_random_LocData(center_lat, center_long, nhouses)
    baseIpAddresses = generate_random_ipAddresses(nhouses)
    fixedPort = 5776

    # Generate each device type for each home
    for houseIndex in range(nhouses):
        loc = locations[houseIndex].lat
        long = locations[houseIndex].long

        network = IPv4Network(baseIpAddresses[houseIndex])
        hosts_iterator = (host for host in network.hosts())

        ips = generate_random_ipAddresses(5)

        tv = IoTDeviceSmartTV(env, ips[0], fixedPort, globalHub, f"{houseIndex}_tv", loc, long)
        brush = IoTDeviceSmartBrush(env, ips[1], fixedPort, globalHub, f"{houseIndex}_brush", loc, long)
        window = IoTDeviceSmartWindow(env, ips[2], fixedPort, globalHub, f"{houseIndex}_window", loc,
                                      long)
        flower = IoTDeviceSmartFlower(env, ips[3], fixedPort, globalHub, f"{houseIndex}_flower", loc,
                                      long)
        kettle = IoTDeviceSmartKettle(env, ips[4], fixedPort, globalHub, f"{houseIndex}_kettle", loc,
                                      long)

        devicesOnThisHouse = [tv, brush, window, flower, kettle]
        allDevices.extend(devicesOnThisHouse)
        allDevicesByHouse[houseIndex] = devicesOnThisHouse

    for houseIndex in range(nhouses):
        allDevicesInThisHouse = allDevicesByHouse[houseIndex]
        for device in allDevicesInThisHouse:
            device.start()


def main():
    # Setup
    random.seed(42)

    # Create environment
    env = simpy.Environment()

    central_lat = 44.42810022576185
    central_long = 26.10414240626916

    # Create central hub
    hub = IoTHub(env, num_loggers=15, num_store_metrics=15, num_rules_processors=2,
                 ip="127.128.23.45", long=str(26.10414240626916), lat=str(44.42810022576185))

    hub.start()

    # Create all houses and devices
    generateRandomDeployment(env,
                             hub,
                             nhouses=NHOUSES,
                             center_lat=central_lat,  # Somewhere Romania
                             center_long=central_long)

    # Create the hacker if needed
    if USE_DEVICE_HACKING:
        hacker = IoTHacker(env, timeToStartHackingAt=START_TIME_HACKING, numDevicesToHack=NUM_DEVICES_TO_HACK)

    # Run the simulation
    env.run(until=46400)#86400)

    hub.resourcesOccupancy.to_csv(f'RESOURCES_OCCUPANCY_HACKED_{USE_DEVICE_HACKING}.csv', header=True, index=False)

    # Save the results
    DATASET_LOGS.to_csv(f'DATASET_LOGS_HACKED_{USE_DEVICE_HACKING}.csv', header=True, index=False)

    DATASET_LOGS_timeout = DATASET_LOGS[DATASET_LOGS['response'] == 0]
    DATASET_LOGS_normal = DATASET_LOGS[DATASET_LOGS['response'] == 200]
    DATASET_LOGS_503 = DATASET_LOGS[DATASET_LOGS['response'] == 503]

    print(f"Database stats: 200:{len(DATASET_LOGS_normal)}, 503:{len(DATASET_LOGS_503)}, 000:{len(DATASET_LOGS_timeout)}")


if __name__ == "__main__":
    main()
