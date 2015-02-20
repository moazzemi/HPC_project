SUBDIRS = src
TOP := $(shell pwd)
SRC = $(TOP)/src/
CC= gcc
#####
#
##
#
#####

all: rosseta

ifeq ($(host-name),)
  host-name := $(shell hostname)
endif

ifeq ($(kernel-name),)
  kernel-name := $(shell uname -s)
endif

ifeq ($(arch-name),)
  arch-name := $(shell uname -p)
endif

ifeq ($(cc-name),)
  cc-name := $(shell basename $(shell which gcc))
  ifeq ($(cc-name),)
    cc-name := $(shell basename $(shell which icc))
  endif
endif
#######RULES######


%.o: %.c
	$(CC) $(CFLAGS_STD) $(DEPSFLAGS) $(CFLAGS) $(COPTFLAGS) -o $@ -c $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS_STD) $(DEPSFLAGS) $(CXXFLAGS) $(CXXOPTFLAGS) -o $@ -c $<
#################

rosseta:$(SRC)/kmeans_rosseta.c 
	$(CC) $(CFLAGS_STD) $(DEPSFLAGS) $(CFLAGS) $(COPTFLAGS)  $(SRC)/kmeans_rosseta.c

  #eof
