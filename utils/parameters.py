#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:32:17 2022
"""

"""
Functions for handling the environmental parameters 
"""

from gym.envs.registration import register

"""
Register all the potential training environments.
"""
def create_env(schedule):
    
    register(
        id='simglucose-child1-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'child#001',
                "schedule":schedule
               }
    )

    register(
        id='simglucose-child2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'child#002',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-child3-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'child#003',
                "schedule":schedule
               }
    )

    register(
        id='simglucose-adolescent1-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#001',
                "schedule":schedule
               }
    )

    register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#002',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-adolescent3-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#003',
                "schedule":schedule
               }
    )

    register(
        id='simglucose-adult1-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#001',
                "schedule":schedule
               }
    )

    register(
        id='simglucose-adult2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#002',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-adult3-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#003',
                "schedule":schedule
               }
    )
    
    
"""
Get the patient parameters.
"""    
def get_params():
    
    params = {
        
        "child#1": {
            
            "folder_name": "child#1",  
            "env_name" : 'simglucose-child1-v0',
            "u2ss" : 1.14220356012,
            "BW" : 34.55648182, 
            "carbohydrate_ratio": 28.6156949676669,
            "correction_factor": 103.016501883601,
            "kp": -1.00E-05,
            "ki": -1.00E-09,
            "kd": -1.00E-03,
            "max_dose": 0.6,
            "replay_name" : "Child#1-1e5"          
            
        },
        
        "child#1-10": {
            
            "folder_name": "child#1",  
            "env_name" : 'simglucose-child1-v0',
            "u2ss" : 1.14220356012,
            "BW" : 34.55648182, 
            "carbohydrate_ratio": 28.6156949676669,
            "correction_factor": 103.016501883601,
            "kp": -1.00E-05,
            "ki": -1.00E-08,
            "kd": -1.00E-03,
            "max_dose": 0.6,
            "replay_name" : "Child#1-1e5-10"          
            
        },
        
        "child#1-20": {
            
            "folder_name": "child#1",  
            "env_name" : 'simglucose-child1-v0',
            "u2ss" : 1.14220356012,
            "BW" : 34.55648182, 
            "carbohydrate_ratio": 28.6156949676669,
            "correction_factor": 103.016501883601,
            "kp": -1.00E-04,
            "ki": -1.00E-07,
            "kd": -1.00E-03,
            "max_dose": 0.6,
            "replay_name" : "Child#1-1e5-20"          
            
        },
        
        "child#2": {
            
            "folder_name": "child#2",            
            "env_name" : 'simglucose-child2-v0',
            "u2ss" : 1.38470169593,
            "BW" : 28.53257352, 
            "carbohydrate_ratio": 27.5060230377229,
            "correction_factor": 99.0216829358025,
            "kp": -1.00E-05,
            "ki": -1.00E-08,
            "kd": -1.00E-02,
            "max_dose": 1.5,
            "replay_name" : "Child#2-1e5"
        
        },
        
        "child#3": {
            
            "folder_name": "child#3",            
            "env_name" : 'simglucose-child3-v0',
            "u2ss" : 0.70038560703,
            "BW" : 41.23304017, 
            "carbohydrate_ratio": 31.2073322051186,
            "correction_factor": 112.346395938427,
            "kp": -1.00E-05,
            "ki": -1.00E-08,
            "kd": -1.00E-03,
            "max_dose": 0.7,
            "replay_name" : "Child#3-1e5"
        
        },
        
        "adolescent#1": {
            
            "folder_name": "adolescent#1",              
            "env_name" : 'simglucose-adolescent1-v0',
            "u2ss" : 1.21697571391,
            "BW" : 68.706, 
            "carbohydrate_ratio": 13.6113998281669,
            "correction_factor": 49.0010393814008,
            "kp": -1.00E-04,
            "ki": -1.00E-07,
            "kd": -1.00E-02,
            "max_dose": 1.5,
            "replay_name" : "Adolescent#1-1e5"
        
        },
        
        "adolescent#1-10": {
            
            "folder_name": "adolescent#1",              
            "env_name" : 'simglucose-adolescent1-v0',
            "u2ss" : 1.21697571391,
            "BW" : 68.706, 
            "carbohydrate_ratio": 13.6113998281669,
            "correction_factor": 49.0010393814008,
            "kp": -1.00E-06,
            "ki": -1.00E-08,
            "kd": -1.00E-02,
            "max_dose": 1.5,
            "replay_name" : "Adolescent#1-1e5-10"
        
        },
        
        "adolescent#1-20": {
            
            "folder_name": "adolescent#1",              
            "env_name" : 'simglucose-adolescent1-v0',
            "u2ss" : 1.21697571391,
            "BW" : 68.706, 
            "carbohydrate_ratio": 13.6113998281669,
            "correction_factor": 49.0010393814008,
            "kp": -1.00E-07,
            "ki": -1.00E-11,
            "kd": -1.00E-02,
            "max_dose": 1.5,
            "replay_name" : "Adolescent#1-1e5-20"
        
        },
        
        "adolescent#2": {
            
            "folder_name": "adolescent#2", 
            "env_name" : 'simglucose-adolescent2-v0',
            "u2ss" : 1.79829979626,
            "BW" : 51.046, 
            "carbohydrate_ratio": 8.06048033285474,
            "correction_factor": 29.0177291982771,
            "kp": -1.00E-04,
            "ki": -1.00E-07,
            "kd": -1.00E-02,
            "max_dose": 1.5,
            "replay_name" : "Adolescent#2-1e5"
        
        },      
        
        "adolescent#3": {
            
            "folder_name": "adolescent#3", 
            "env_name" : 'simglucose-adolescent3-v0',
            "u2ss" : 1.4462660088,
            "BW" : 44.791, 
            "carbohydrate_ratio": 20.6246970212749,
            "correction_factor": 74.2489092765897,
            "kp": -1.00E-04,
            "ki": -1.00E-07,
            "kd": -1.00E-02,
            "max_dose": 1.2,
            "replay_name" : "Adolescent#3-1e5"
        
        },  
        
        "adult#1": {
            
            "folder_name": "adult#1",             
            "env_name" : 'simglucose-adult1-v0',
            "u2ss" : 1.2386244136,
            "BW" : 102.32, 
            "carbohydrate_ratio": 9.9173582569505,
            "correction_factor": 35.7024897250218,
            "kp": -1.00E-04,
            "ki": -1.00E-07,
            "kd": -1.00E-02,
            "max_dose": 0.75,
            "replay_name" : "Adult#1-1e5"
        
        }, 
        
        "adult#1-10": {
            
            "folder_name": "adult#1",             
            "env_name" : 'simglucose-adult1-v0',
            "u2ss" : 1.2386244136,
            "BW" : 102.32, 
            "carbohydrate_ratio": 9.9173582569505,
            "correction_factor": 35.7024897250218,
            "kp": -1.00E-06,
            "ki": -1.00E-08,
            "kd": -1.00E-02,
            "max_dose": 0.75,
            "replay_name" : "Adult#1-1e5-10"
        
        }, 
        
        "adult#1-20": {
            
            "folder_name": "adult#1",             
            "env_name" : 'simglucose-adult1-v0',
            "u2ss" : 1.2386244136,
            "BW" : 102.32, 
            "carbohydrate_ratio": 9.9173582569505,
            "correction_factor": 35.7024897250218,
            "kp": -1.00E-07,
            "ki": -1.00E-11,
            "kd": -1.00E-02,
            "max_dose": 0.75,
            "replay_name" : "Adult#1-1e5-20"
        
        }, 
        
        "adult#2": {
            
            "folder_name": "adult#2",   
            "env_name" : 'simglucose-adult2-v0',
            "u2ss" : 1.23270240324,
            "BW" : 111.1, 
            "carbohydrate_ratio": 8.64023791338857,
            "correction_factor": 31.1048564881989,
            "kp": -1.00E-04,
            "ki": -1.00E-07,
            "kd": -1.00E-02,
            "max_dose": 0.7, 
            "replay_name" : "Adult#2-1e5"
        
        },
        
        "adult#3": {
            
            "folder_name": "adult#3",   
            "env_name" : 'simglucose-adult3-v0',
            "u2ss" : 1.74604298612,
            "BW" : 81.631, 
            "carbohydrate_ratio": 8.86057935797141,
            "correction_factor": 31.8980856886971,
            "kp": -1.00E-04,
            "ki": -1.00E-07,
            "kd": -1.00E-02,
            "max_dose": 0.9, 
            "replay_name" : "Adult#3-1e5"
        
        } 
    
    }
    
    return params