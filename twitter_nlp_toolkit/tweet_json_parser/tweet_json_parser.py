#!/usr/bin/python
# -*- coding: utf-8 -*-
# twitter sends its data as a json

import json

# we will convert the json data into a csv

import csv

# numpy's np.nan helps us deal with missing values (and there is a TON of missing values in tweets)

import numpy as np

import pandas as pd


class json_parser:

    def parse_json_file_into_dataframe(self, json_file_name):

        # The data is a newline separated list of json objects

        jsonList = []

        # Read the contents of the file to a string

        with open(json_file_name, 'r', encoding='latin-1') as file:
            data = file.read()

        # Make a json object from that string (the data is a newline delimited list of jsons)

        for line in data.splitlines():
            jsonList.append(json.loads(line))

        result_df = pd.json_normalize(jsonList)  # flatten the json and convert it to df

        return result_df

    def parse_json_file_into_csv(self, json_file_name,
                                 output_file_name):

        # The data is a newline separated list of json objects

        jsonList = []

        # Read the contents of the file to a string

        with open(json_file_name, 'r', encoding='latin-1') as file:
            data = file.read()

        # Make a json object from that string (the data is a newline delimited list of jsons)

        for line in data.splitlines():
            jsonList.append(json.loads(line))

        result_df = pd.json_normalize(jsonList)  # flatten the json and convert it to df

        result_df.to_csv(output_file_name)