#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:29:03 2022
"""

"""
Functions for general use across data collection, training and evaluation. 
"""

import math
import numpy as np


"""
When provided with a blood glucose output from UVA/Padova simulator it 
calculates the corresponding magni risk and returns as a floats. 
"""
def calculate_risk(blood_glucose):
    return 10 * math.pow((3.5506 * (math.pow(math.log(max(1, blood_glucose[0])), 0.8353) - 3.7932)), 2)    
    
"""
Uses the current blood glucose value, meal history and current meals carbs
to calculate the optimal bolus dose for a meal for a patient
"""    
def calculate_bolus(blood_glucose, meal_history, current_meal, 
                    carbohyrdate_ratio, correction_factor, target_blood_glucose):   

    # calculate the meal bolus using meal carbs
    bolus = current_meal / carbohyrdate_ratio
    
    # if a meal hasn't occurred in meal history
    if np.sum(meal_history) == 0: 
        
        # correct the bolus for high or low blood glucose
        bolus += (blood_glucose[0] - target_blood_glucose) / correction_factor
        
    return bolus / 3

"""
When given the current blood glucose value determine if it falls in range and 
return a value indicating its position.
"""    
def is_in_range(blood_glucose, hypo_threshold, hyper_threshold, sig_hypo_threshold, sig_hyper_threshold):
    
        # output: 0 = in range, 1 = hyper, -1 = hypo 
        if blood_glucose > sig_hyper_threshold: return 2     
        elif blood_glucose > hyper_threshold: return 1
        elif blood_glucose < sig_hypo_threshold: return -2   
        elif blood_glucose < hypo_threshold: return -1
        else: return 0     
  
"""
Calculate the recommended basal dose for a patient based on their current 
blood glucose and their parameters.
"""            
def PID_action(blood_glucose, previous_error, integrated_state, 
               target_blood_glucose, kp, ki, kd, basal_default):
    
    # proportional control
    error = target_blood_glucose - blood_glucose[0] 
    p_act = kp * error
    
    # integral control        
    integrated_state += error
    i_act = ki * integrated_state
    
    # derivative control
    d_act = kd * (error - previous_error)
    previous_error = error
    
    # get the final dose output
    calculated_dose = np.array([(p_act + i_act + d_act + basal_default) / 3], dtype=np.float32)
    
    return calculated_dose, previous_error, integrated_state

