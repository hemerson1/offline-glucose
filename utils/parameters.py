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
        id='simglucose-child4-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'child#004',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-child5-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'child#005',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-child6-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'child#006',
                "schedule":schedule
               }
    )

    register(
        id='simglucose-child7-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'child#007',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-child8-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'child#008',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-child9-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'child#009',
                "schedule":schedule
               }
    )

    register(
        id='simglucose-child10-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'child#010',
                "schedule":schedule
               }
    )
    
    # ADOLESCENTS #######################################

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
        id='simglucose-adolescent4-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#004',
                "schedule":schedule
               }
    )

    register(
        id='simglucose-adolescent5-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#005',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-adolescent6-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#006',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-adolescent7-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#007',
                "schedule":schedule
               }
    )

    register(
        id='simglucose-adolescent8-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#008',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-adolescent9-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#009',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-adolescent10-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#010',
                "schedule":schedule
               }
    )
    
    # ADULTS ##################################################

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
    
    register(
        id='simglucose-adult4-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#004',
                "schedule":schedule
               }
    )

    register(
        id='simglucose-adult5-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#005',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-adult6-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#006',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-adult7-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#007',
                "schedule":schedule
               }
    )

    register(
        id='simglucose-adult8-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#008',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-adult9-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#009',
                "schedule":schedule
               }
    )
    
    register(
        id='simglucose-adult10-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#010',
                "schedule":schedule
               }
    )

    
    
"""
Get the patient parameters.
"""    
def get_params():
    
    params = {
        
        # CHILDREN ##############################################
        
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
        
        "child#4": {
            
            "folder_name": "child#4",            
            "env_name" : 'simglucose-child4-v0',
            "u2ss" : 1.38610897835,
            "BW" : 35.5165043, 
            "carbohydrate_ratio": 25.23323213020198,
            "correction_factor": 90.83963566872713,
            "kp": -1e-05,
            "ki": -1e-11,
            "kd": -1e-03,
            "max_dose": 0.5,
            "replay_name" : "Child#4-1e5"
        
        },
        
        "child#5": {
            
            "folder_name": "child#5",            
            "env_name" : 'simglucose-child5-v0',
            "u2ss" : 1.36318862639,
            "BW" : 37.78855797, 
            "carbohydrate_ratio": 12.21462592173681,
            "correction_factor": 43.97265331825251,
            "kp": -1e-04,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.5,
            "replay_name" : "Child#5-1e5"
        
        },
        
        "child#6": {
            
            "folder_name": "child#6",            
            "env_name" : 'simglucose-child6-v0',
            "u2ss" : 0.985487128297,
            "BW" : 41.00214896, 
            "carbohydrate_ratio": 24.723079998277314,
            "correction_factor": 89.00308799379833,
            "kp": -1e-05,
            "ki": -1e-08,
            "kd": -1e-03,
            "max_dose": 0.5,
            "replay_name" : "Child#6-1e5"
        
        },
        
        "child#7": {
            
            "folder_name": "child#7",            
            "env_name" : 'simglucose-child7-v0',
            "u2ss" : 1.02592147609,
            "BW" : 45.5397665, 
            "carbohydrate_ratio": 13.807252026084589,
            "correction_factor": 49.706107293904516,
            "kp": -1e-07,
            "ki": -1e-08,
            "kd": -1e-03,
            "max_dose": 0.5,
            "replay_name" : "Child#7-1e5"
        
        },
        
        "child#8": {
            
            "folder_name": "child#8",            
            "env_name" : 'simglucose-child8-v0',
            "u2ss" : 1.43273282863,
            "BW" : 23.73405728, 
            "carbohydrate_ratio": 23.261842061321445,
            "correction_factor": 83.7426314207572,
            "kp": -1e-07,
            "ki": -1e-11,
            "kd": -1e-02,
            "max_dose": 5.0,
            "replay_name" : "Child#8-1e5"
        
        },
        
        "child#9": {
            
            "folder_name": "child#9",            
            "env_name" : 'simglucose-child9-v0',
            "u2ss" : 1.10155422738,
            "BW" : 35.53392558, 
            "carbohydrate_ratio": 28.74519570209282,
            "correction_factor": 103.48270452753414,
            "kp": -1e-07,
            "ki": -1e-07,
            "kd": -1e-07,
            "max_dose": 2.5,
            "replay_name" : "Child#9-1e5"
        
        },
        
        "child#10": {
            
            "folder_name": "child#10",            
            "env_name" : 'simglucose-child10-v0',
            "u2ss" : 1.12891185261,
            "BW" : 35.21305847, 
            "carbohydrate_ratio": 24.21108601288932,
            "correction_factor": 87.15990964640156,
            "kp": -1e-05,
            "ki": -1e-08,
            "kd": -1e-03,
            "max_dose": 0.5,
            "replay_name" : "Child#10-1e5"
        
        }, 
        
        # ADOLESCENTS #############################################
        
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
        
        "adolescent#4": {
            
            "folder_name": "adolescent#4", 
            "env_name" : 'simglucose-adolescent4-v0',
            "u2ss" : 1.76263284642,
            "BW" : 49.564, 
            "carbohydrate_ratio": 14.18324377702899,
            "correction_factor": 51.05967759730436,
            "kp": -1e-04,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.5,
            "replay_name" : "Adolescent#4-1e5"
        
        },      
        
        "adolescent#5": {
            
            "folder_name": "adolescent#5", 
            "env_name" : 'simglucose-adolescent5-v0',
            "u2ss" : 1.5346452819,
            "BW" : 47.074, 
            "carbohydrate_ratio": 14.703840790944376,
            "correction_factor": 52.93382684739976,
            "kp": -1e-04,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.5,
            "replay_name" : "Adolescent#5-1e5"
        
        },
        "adolescent#6": {
            
            "folder_name": "adolescent#6", 
            "env_name" : 'simglucose-adolescent6-v0',
            "u2ss" : 1.92787834743,
            "BW" : 45.408, 
            "carbohydrate_ratio": 10.084448671441356,
            "correction_factor": 36.30401521718888,
            "kp": -1e-04,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.5,
            "replay_name" : "Adolescent#6-1e5"
        
        },      
        
        "adolescent#7": {
            
            "folder_name": "adolescent#7", 
            "env_name" : 'simglucose-adolescent7-v0',
            "u2ss" : 2.04914771228,
            "BW" : 37.898, 
            "carbohydrate_ratio": 11.457886857675446,
            "correction_factor": 41.24839268763161,
            "kp": -1e-04,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.75,
            "replay_name" : "Adolescent#7-1e5"
        
        },
        "adolescent#8": {
            
            "folder_name": "adolescent#8", 
            "env_name" : 'simglucose-adolescent8-v0',
            "u2ss" : 1.35324144985,
            "BW" : 41.218, 
            "carbohydrate_ratio": 7.888090404486432,
            "correction_factor": 28.397125456151155,
            "kp": -1e-04,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 2.5,
            "replay_name" : "Adolescent#8-1e5"
        
        },      
        
        "adolescent#9": {
            
            "folder_name": "adolescent#9", 
            "env_name" : 'simglucose-adolescent9-v0',
            "u2ss" : 1.38186522046,
            "BW" : 43.885, 
            "carbohydrate_ratio": 20.76570050945875,
            "correction_factor": 74.7565218340515,
            "kp": -1e-07,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.5,
            "replay_name" : "Adolescent#9-1e5"
        
        },
        
        "adolescent#10": {
            
            "folder_name": "adolescent#10", 
            "env_name" : 'simglucose-adolescent10-v0',
            "u2ss" : 1.66109036262,
            "BW" : 47.378, 
            "carbohydrate_ratio": 15.07226804643741,
            "correction_factor": 54.260164967174674,
            "kp": -1e-04,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.5,
            "replay_name" : "Adolescent#10-1e5"        
        },              
        
        # ADULTS ###############################################
        
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
        
        },
        
        "adult#4": {
            
            "folder_name": "adult#4",   
            "env_name" : 'simglucose-adult4-v0',
            "u2ss" : 1.40925544793,
            "BW" : 63.0, 
            "carbohydrate_ratio": 14.789424168083986,
            "correction_factor": 53.24192700510235,
            "kp": -1e-05,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.5, 
            "replay_name" : "Adult#4-1e5"
        
        },
        
        "adult#5": {
            
            "folder_name": "adult#5",   
            "env_name" : 'simglucose-adult5-v0',
            "u2ss" : 1.25415109169,
            "BW" : 94.074, 
            "carbohydrate_ratio": 7.318937998432252,
            "correction_factor": 26.348176794356107,
            "kp": -1e-03,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.0, 
            "replay_name" : "Adult#5-1e5"
        
        },
        
        "adult#6": {
            
            "folder_name": "adult#6",   
            "env_name" : 'simglucose-adult6-v0',
            "u2ss" : 2.60909529933,
            "BW" : 66.097, 
            "carbohydrate_ratio": 8.144806942246657,
            "correction_factor": 29.321304992087967,
            "kp": -1e-04,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.5, 
            "replay_name" : "Adult#6-1e5"
        
        },
        
        "adult#7": {
            
            "folder_name": "adult#7",   
            "env_name" : 'simglucose-adult7-v0',
            "u2ss" : 1.50334589878,
            "BW" : 91.229, 
            "carbohydrate_ratio": 11.902889350456292,
            "correction_factor": 42.85040166164265,
            "kp": -1e-07,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.0, 
            "replay_name" : "Adult#7-1e5"
        
        },
        
        "adult#8": {
            
            "folder_name": "adult#8",   
            "env_name" : 'simglucose-adult8-v0',
            "u2ss" : 1.11044245549,
            "BW" : 102.79, 
            "carbohydrate_ratio": 11.68803605523481,
            "correction_factor": 42.07692979884532,
            "kp": -1e-04,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.0, 
            "replay_name" : "Adult#8-1e5"
        
        },
        
        "adult#9": {
            
            "folder_name": "adult#9",   
            "env_name" : 'simglucose-adult9-v0',
            "u2ss" : 1.51977345451,
            "BW" : 74.604, 
            "carbohydrate_ratio": 7.439205003922471,
            "correction_factor": 26.781138014120895,
            "kp": -1e-04,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.5, 
            "replay_name" : "Adult#9-1e5"
        
        },
        
        "adult#10": {
            
            "folder_name": "adult#10",   
            "env_name" : 'simglucose-adult10-v0',
            "u2ss" : 1.37923535927,
            "BW" : 73.859, 
            "carbohydrate_ratio": 7.758126846037283,
            "correction_factor": 27.92925664573422,
            "kp": -1e-04,
            "ki": -1e-07,
            "kd": -1e-02,
            "max_dose": 1.5, 
            "replay_name" : "Adult#10-1e5"
        
        },
        
        
    
    }
    
    return params
