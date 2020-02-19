#!/bin/bash

rm *.tmp
rm *.zip
rm *.csv
rm *.pkl

rm -rf trained_models/

rm -rf twitter_nlp_toolkit.egg-info

if [ -d build ]; then rm -rf build; fi 

if [ -d dist ]; then rm -rf dist; fi 
