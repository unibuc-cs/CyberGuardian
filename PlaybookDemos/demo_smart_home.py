# The playbook is a sequence of steps that the user can follow to solve a problem. The assistant will guide the user through the steps.
# Smart Home demo
import streamlit as st
import time

from PlaybookDemos.commons import SLEEP_TIME_BETWEEN_QUESTIONS

## Case 1: smart home
def demo_trigger_msg_SmartHome():
    st.session_state.messages.append({"role": "assistant",
                                      "content": "Alert: there seems to be many timeouts and 503 error codes on the IoT Hub. Please investigate! I can help you with this."})

def doDemoScript_SmartHome():
    if st.session_state.DEBUG_SKIP_TO_STEP is not None:
        st.session_state.DEMO_MODE_STEP = st.session_state.DEBUG_SKIP_TO_STEP
        st.session_state.DEBUG_SKIP_TO_STEP = None

    if st.session_state.DEMO_MODE_STEP == 1:
        st.session_state.messages.append({"role": "user",
                                          "content": "Ok. I'm on it, can you show me a resource utilization graph comparison between a normal session and current situation"})
        st.session_state.DEMO_MODE_STEP += 1
        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()
    elif st.session_state.DEMO_MODE_STEP == 2:
        st.session_state.messages.append({"role": "user",
                                          "content": "Show me the logs of the devices grouped by IP which have more than 25% requests over the median of a normal session per. Sort them by count"})
        st.session_state.DEMO_MODE_STEP += 1
        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()
    elif st.session_state.DEMO_MODE_STEP == 3:
        st.session_state.messages.append({"role": "user",
                                          "content": "Can you show a sample of GET requests from the top 3 demanding IPs, including their start time, end time? Only show the last 10 logs."})
        st.session_state.DEMO_MODE_STEP += 1
        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()
    elif st.session_state.DEMO_MODE_STEP == 4:
        st.session_state.messages.append({"role": "user",
                                          "content": "Give me a world map of requests by comparing the current Data and a known snapshot with bars"})
        st.session_state.DEMO_MODE_STEP += 1
        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()
    elif st.session_state.DEMO_MODE_STEP == 5:
        st.session_state.messages.append({"role": "user",
                                          "content": "What could it mean if there are many IPs from different locations sending GET commands in a short time with random queries ?"})
        st.session_state.DEMO_MODE_STEP += 1
        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()
    elif st.session_state.DEMO_MODE_STEP == 6:
        st.session_state.messages.append({"role": "user",
                                          "content": "Generate and execute a python code to insert in the Firewall dataset IP 130.112.80.168 as blocked  which does not seem legit"}) # these top IPs excepting 130.112.80.168 which seems like a legit one."})
        st.session_state.DEMO_MODE_STEP += 1
        time.sleep(SLEEP_TIME_BETWEEN_QUESTIONS)
        st.rerun()
