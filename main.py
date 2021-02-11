# from typing import Optional
import datetime
from tensorflow import keras
import pickle

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Laps Prediction": "Version 01"}


@app.get("/laps_predict")
def read_item(Premium: float, BSA: float, SumAtRisk: float, Monthly_Income: float, TotalTermYrs: int, Plan: str, Sex: str, Birth_Yr: int, Pre_Policies: int, Beneficiary_Count: int, Spouse_yesno: str, Field_Agent: str, Agent_Start_Year: int, Agent_INFC_Count: int, Agent_ALAP_Count: int, Agent_TLAP_Count: int):

    ######################################
    ############### Mod 1 ################
    ######################################

    # load model and scaler
    mod1 = keras.models.load_model('mod/ModelFirst.h5')
    sc1 = pickle.load(open('mod/ScalerFirst.pkl', 'rb'))

    now = datetime.datetime.now()
    mod1_input = []

    # feature 1
    premium = Premium
    mod1_input.append(premium)

    # feature 2
    bsa = BSA
    mod1_input.append(bsa)

    # feature 3
    sumatrisk = SumAtRisk
    mod1_input.append(sumatrisk)

    # feature 4
    totalterms = TotalTermYrs*12
    mod1_input.append(totalterms)

    # feature 5
    pre_policies = Pre_Policies
    mod1_input.append(pre_policies)

    # feature 6
    beneficiary_count = Beneficiary_Count
    mod1_input.append(beneficiary_count)

    # feature 7
    if Spouse_yesno == 'Yes':
        spouse_yesno = 1
    else:
        spouse_yesno = 0
    mod1_input.append(spouse_yesno)

    # feature 8
    if beneficiary_count>0:
        ceneficiary_count_cat = 1
    else:
        ceneficiary_count_cat = 0
    mod1_input.append(ceneficiary_count_cat)

    # feature 9
    if Field_Agent == 'Yes':
        agentuw = 1
    else:
        agentuw = 0
    mod1_input.append(agentuw)

    # feature 10
    agent_start_year = Agent_Start_Year
    mod1_input.append(agent_start_year)

    # feature 11
    agent_exp = now.year - agent_start_year
    mod1_input.append(agent_exp)

    # feature 12
    age = now.year - Birth_Yr
    mod1_input.append(age)

    # feature 13
    infc_count = Agent_INFC_Count
    mod1_input.append(infc_count)

    # feature 14
    alap_count = Agent_ALAP_Count
    mod1_input.append(alap_count)

    # feature 15
    tlap_count = Agent_TLAP_Count
    mod1_input.append(tlap_count)


    # feature 16 17 18
    if Plan == 'FMLY':
        plan_list = [0, 0, 0]
    elif Plan == 'PRP2':
        plan_list = [0, 1, 0]
    elif Plan == 'SLUN':
        plan_list = [0, 0, 1]
    else:
        plan_list = [1, 0, 0]
    mod1_input = mod1_input + plan_list

    # feature 19
    if Sex == 'Male':
        Gender = 1
    else:
        Gender = 0
    mod1_input.append(Gender)

    # feature 20
    month = now.month
    mod1_input.append(month)

    # mod 1 features
    mod1_input = [mod1_input]

    # predicting
    in_data1 = sc1.transform(mod1_input)
    out1 = mod1.predict_classes(in_data1)

    ######################################
    ############### Mod 2 ################
    ######################################

    # load model and scaler
    mod2 = keras.models.load_model('mod/ModelSec.h5')
    sc2 = pickle.load(open('mod/ScalerSec.pkl', 'rb'))

    mod2_input = []

    # feature 1
    mod2_input.append(premium)

    # feature 2
    mod2_input.append(bsa)

    # feature 3
    mod2_input.append(TotalTermYrs)

    # feature 4
    mod2_input.append(sumatrisk)

    # feature 5
    mod2_input.append(now.year)

    # feature 6
    mod2_input.append(Monthly_Income)

    # feature 7
    mod2_input.append(Gender)

    # feature 8
    mod2_input.append(agent_exp)

    # feature 9
    mod2_input.append(age)

    # feature 10
    mod2_input.append(infc_count)

    # feature 11
    mod2_input.append(alap_count)

    # feature 12
    mod2_input.append(tlap_count)


    # feature 13, 14, 15, 16, 17
    if Plan == 'AAPP':
        pl_list = [0, 0, 0, 0, 0]
    elif Plan == 'CHLD':
        pl_list = [1, 0, 0, 0, 0]
    elif Plan == 'FMLY':
        pl_list = [0, 1, 0, 0, 0]
    elif Plan == 'PRP2':
        pl_list = [0, 0, 0, 1, 0]
    elif Plan == 'SLUN':
        pl_list = [0, 0, 0, 0, 1]
    else:
        pl_list = [0, 0, 1, 0, 0]

    mod2_input = mod2_input + pl_list

    # feature 18
    mod2_input.append(agentuw)

    # feature 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
    if month == 1:
        month_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif month == 2:
        month_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif month == 3:
        month_list = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif month == 4:
        month_list = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif month == 5:
        month_list = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif month == 6:
        month_list = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif month == 7:
        month_list = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif month == 8:
        month_list = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif month == 9:
        month_list = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif month == 10:
        month_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif month == 11:
        month_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    else:
        month_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    mod2_input = mod2_input + month_list

    # mod 2 features
    mod2_input = [mod2_input]

    # predicting
    in_data2 = sc2.transform(mod2_input)
    out2 = mod2.predict_classes(in_data2)

    if ((int(out1[0][0]) == 1) & (int(out2[0][0]) == 0)):
        out_var = 'High'
    elif ((int(out1[0][0]) == 0) & (int(out2[0][0]) == 1)):
        out_var = 'Low'
    else:
        out_var = 'Medium'


    return {"Risk": out_var}
