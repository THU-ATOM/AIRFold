from datetime import datetime
from enum import Enum, unique


@unique
class State(Enum):
    UNKNOWN = 0

    AIRFOLD_START = 1

    RECEIVED = 100
    RECEIVE_ERROR = 910
    POST_RECEIVE = 101  # receive stage

    SEARCH_START = 200

    HHBLITS_START = 201
    HHBLITS_SUCCESS = 202
    HHBLITS_ERROR = 920

    JACKHMM_START = 203
    JACKHMM_SUCCESS = 204
    JACKHMM_ERROR = 921
    
    BLAST_START = 205
    BLAST_SUCCESS = 206
    BLAST_ERROR = 922

    MMSEQS_START = 207
    MMSEQS_SUCCESS = 208
    MMSEQS_ERROR = 923

    DEEPqMSA_START = 209
    DEEPqMSA_SUCCESS = 210
    DEEPqMSA_ERROR = 924
    DEEPdMSA_START = 211
    DEEPdMSA_SUCCESS = 212
    DEEPdMSA_ERROR = 925
    DEEPmMSA_START = 213
    DEEPmMSA_SUCCESS = 214
    DEEPmMSA_ERROR = 926


    SEARCH_SUCCESS = 215
    SEARCH_ERROR = 927  # msa search stage

    CLUSTE_START = 216
    CLUSTE_SUCCESS = 217
    CLUSTE_ERROR = 928

    SELECT_START = 220
    SELECT_SUCCESS = 221
    SELECT_ERROR = 929

    TPLT_SEARCH_START = 230
    TPLT_SEARCH_SUCCESS = 231
    TPLT_SEARCH_ERROR = 940

    TPLT_SELECT_START = 232
    TPLT_SELECT_SUCCESS = 233
    TPLT_SELECT_ERROR = 941

    TPLT_FEAT_START = 240
    TPLT_FEAT_SUCCESS = 241
    TPLT_FEAT_ERROR = 942

    MSA2FEATURE_START = 300
    MSA2FEATURE_SUCCESS = 301
    MSA2FEATURE_ERROR = 930

    STRUCTURE_START = 310
    STRUCTURE_SUCCESS = 311
    STRUCTURE_ERROR = 931

    RELAX_START = 320
    RELAX_SUCCESS = 321
    RELAX_ERROR = 932

    ANALYSIS_GEN = 380

    SUBMIT_START = 400
    SUBMIT_SUCCESS = 401
    SUBMIT_SKIP = 402

    SUBMIT_ERROR = 991

    PREPROCESS_START = 500
    PREPROCESS_SUCCESS = 501
    PREPROCESS_ERROR = 950

    KILLED = 998
    RUNTIME_ERROR = 999

    ERROR = 900


stateName2code = {s.name: s.value for s in State}
stateCode2name = {s.value: s.name for s in State}
state2code = {s: s.value for s in State}
code2state = {s.value: s for s in State}


def is_failed(code: State):
    return code.value >= State.ERROR.value


def is_success(code: State):
    return code.value < State.ERROR.value


def get_state2message(code: State):
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{code.name}:{time}"
