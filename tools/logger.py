# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/1/7 15:24
'''
import logging

logger = logging.getLogger(__name__)


def setting_logging(task_name):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logger.setLevel(logging.DEBUG)

    f_handler = logging.FileHandler("logs/{}.txt".format(task_name))
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(f_handler)
    logger.addHandler(console_handler)


# def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
#     total_parameters = 0
#     parameters_string = ""

#     for variable in tf.trainable_variables():

#         shape = variable.get_shape()
#         variable_parameters = 1
#         for dim in shape:
#             variable_parameters *= dim.value
#         total_parameters += variable_parameters
#         if len(shape) == 1:
#             parameters_string += ("%s %d, " % (variable.name, variable_parameters))
#         else:
#             parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

#         logger.info(parameters_string)
#         logger.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
