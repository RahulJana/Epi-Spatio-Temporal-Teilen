# -*- coding: utf-8 -*-

import os
from datetime import datetime

CURRENT_TIME = datetime.now().strftime("%Y%m%d%H%M%S")


class Paths:
    MOCKED = os.path.join("mock_data", "{data_type}_epidata", "covid")
    PROCESSED = os.path.join("report_data", "{data_type}_epidata", "covid", "{location_type}", "processed")

    CUMUCONFIRMED = os.path.join(PROCESSED, "cumuConfirmed.csv")
    CUMURECOVERED = os.path.join(PROCESSED, "cumuRecovered.csv")
    CUMUDEATH = os.path.join(PROCESSED, "cumuDeath.csv")
    POPULATION = os.path.join(PROCESSED, "population.csv")

    ROOT_DIR = "Spatial-SIR-C_GNNs"
    DATA_RAW = os.path.join(ROOT_DIR, "raw_data")
    DATA_REPO = os.path.join(ROOT_DIR, "repo_data")
    
    HM = os.path.join(DATA_RAW, "report_data", "{data_type}_epidata", "mobility")
    PROVINCE_INDEX = os.path.join(HM, "provinces.json")
    CITY_INDEX = os.path.join(HM, "cities.json")
    NEIGHBOR_ADJACENCY_MATRIX = os.path.join(HM, "neighbor_adjacency_matrix.csv")

    RESULT = os.path.join(ROOT_DIR, "result")
    OVERCOME = os.path.join(ROOT_DIR, "overcome")
    TEST = os.path.join(ROOT_DIR, "test")

    HM_DIAGRAM = os.path.join(ROOT_DIR, "diagram", "hm_diagram")
    EPI_DIAGRAM = os.path.join(ROOT_DIR, "diagram", "epi_diagram")

    DEST_DIR = os.path.join(TEST, CURRENT_TIME)
    
    os.makedirs(DATA_REPO, exist_ok=True)
    os.makedirs(HM_DIAGRAM, exist_ok=True)
    os.makedirs(EPI_DIAGRAM, exist_ok=True)
    os.makedirs(DEST_DIR, exist_ok=True)


class NAME:
    LOG_NAME = "ssirstgnn.log"


class DATE:
    START_DATE, END_DATE = "20211030", "20221030"  #  "20200503" # "20200122", "20221030"
    DATE_SELECTED = f"{START_DATE}_{END_DATE}"
    DATE_LIST = []


class Color:
    COLORS = [
        "green",
        "red",
        "black",
        "orange",
        "yellow",
        "purple",
        "pink",
        "brown",
        "grey",
        "cyan",
        "magenta",
        "blue",
        "lime",
        "teal",
        "navy",
        "maroon",
        "olive",
        "silver",
        "gold",
        "peachpuff",
        "darkorange",
        "seagreen",
        "darkviolet",
        "lightpink",
    ]

