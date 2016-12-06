#!/bin/bash

g++ main.cpp `pkg-config --cflags --libs opencv` -o main 
