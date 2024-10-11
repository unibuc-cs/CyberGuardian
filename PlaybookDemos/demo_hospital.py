# The playbook is a sequence of steps that the user can follow to solve a problem. The assistant will guide the user through the steps.
# Hospital demo
from PlaybookDemos.commons import SLEEP_TIME_BETWEEN_QUESTIONS
import streamlit as st
import time

## Case 2: Hospital IT

def demo_trigger_msg_Hospital():
    st.session_state.messages.append({"role": "assistant",
                                      "content": "Alert: there are many issues opened in the ticketing system suggesting "
                                                 "that that doctors can't access the patients' DICOM and X-Ray records. "
                                                 "Please investigate! I can help you with this."})

    st.session_state.DEMO_MODE_STEP = 1 # Reset the step

def doDemoScript_Hospital():
    if st.session_state.DEBUG_SKIP_TO_STEP is not None:
        st.session_state.DEMO_MODE_STEP = st.session_state.DEBUG_SKIP_TO_STEP
        st.session_state.DEBUG_SKIP_TO_STEP = None

    if st.session_state.DEMO_MODE_STEP == 1:
        st.session_state.messages.append({"role": "user",
                                          "content": "What are the IPs of the servers hosting the DICOM "
                                                     "and X-Ray records? Can you show me a graph "
                                                     "of their resource utilization over the last 24 hours?"})
        st.session_state.DEMO_MODE_STEP += 1
        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()
    elif st.session_state.DEMO_MODE_STEP == 2:
        # st.session_state.messages.append({"role": "user",
        #                                   "content": "Can you show the logs of internal servers handling these services "
        #                                              "grouped by IP which have more than 35% requests over "
        #                                              "the median of a normal session per. Sort them by count"})
        st.session_state.DEMO_MODE_STEP += 1
        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()
    elif st.session_state.DEMO_MODE_STEP == 3:
        st.session_state.messages.append({"role": "user",
                                          "content": "Give me a map with locations where these requests come from by comparing the current "
                                                     "requests and a normal day usage, using bars and colors"})
        st.session_state.DEMO_MODE_STEP += 1
        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()
    elif st.session_state.DEMO_MODE_STEP == 4:
        st.session_state.messages.append({"role": "user",
                                          "content": "Can you show a sample of GET requests from the top 10 demanding IPs, highlighting the first 4?"
                                                     " Include their IDs, locations, and number of requests."})
        st.session_state.DEMO_MODE_STEP += 1

        if st.session_state.DEBUG_SKIP_JUMP_TO_NEXT:
            st.session_state.DEMO_MODE_STEP = st.session_state.DEBUG_SKIP_JUMP_TO_NEXT
            st.session_state.DEBUG_SKIP_JUMP_TO_NEXT = None

        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()

    elif st.session_state.DEMO_MODE_STEP == 5:
        st.session_state.messages.append({"role": "user",
                                          "content": "Can it be an attack if several servers receive too many queries from different IPs at random locations in a very short time window?"})
        st.session_state.DEMO_MODE_STEP += 1
        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()
    elif st.session_state.DEMO_MODE_STEP == 6:
        st.session_state.messages.append({"role": "user",
                                          "content": "Generate and execute a python code to insert in the Firewall dataset IP 130.112.80.168 as blocked which does not seem legit"}) #excepting 130.112.80.168 which seems like a legit one."})
        st.session_state.DEMO_MODE_STEP += 1
        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()
        ##########################################
