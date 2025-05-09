# mental_health_questionnaire.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

class MentalHealthQuestionnaire:
    """
    A questionnaire module that administers psychological assessments
    and provides scores and interpretations for the EmotionAgent.
    """
    
    def __init__(self):
        # Define the questionnaires with their questions, scoring methods, and interpretations
        self.questionnaires = {
            "mood": {
                "name": "Mood Assessment (PHQ-9 Modified)",
                "description": "Evaluates symptoms of depression over the past two weeks",
                "questions": [
                    "Little interest or pleasure in doing things",
                    "Feeling down, depressed, or hopeless",
                    "Trouble falling or staying asleep, or sleeping too much",
                    "Feeling tired or having little energy",
                    "Poor appetite or overeating",
                    "Feeling bad about yourself or that you are a failure",
                    "Trouble concentrating on things",
                    "Moving or speaking slowly, or being fidgety/restless",
                    "Thoughts that you would be better off dead or of hurting yourself"
                ],
                "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
                "scores": [0, 1, 2, 3],
                "interpretation": {
                    "0-4": {"level": "Minimal", "description": "Minimal or no depression"},
                    "5-9": {"level": "Mild", "description": "Mild depression"},
                    "10-14": {"level": "Moderate", "description": "Moderate depression"},
                    "15-19": {"level": "Moderately Severe", "description": "Moderately severe depression"},
                    "20-27": {"level": "Severe", "description": "Severe depression"}
                }
            },
            "anxiety": {
                "name": "Anxiety Assessment (GAD-7 Modified)",
                "description": "Measures anxiety symptoms over the past two weeks",
                "questions": [
                    "Feeling nervous, anxious, or on edge",
                    "Not being able to stop or control worrying",
                    "Worrying too much about different things",
                    "Trouble relaxing",
                    "Being so restless that it's hard to sit still",
                    "Becoming easily annoyed or irritable",
                    "Feeling afraid as if something awful might happen"
                ],
                "options": ["Not at all", "Several days", "More than half the days", "Nearly every day"],
                "scores": [0, 1, 2, 3],
                "interpretation": {
                    "0-4": {"level": "Minimal", "description": "Minimal anxiety"},
                    "5-9": {"level": "Mild", "description": "Mild anxiety"},
                    "10-14": {"level": "Moderate", "description": "Moderate anxiety"},
                    "15-21": {"level": "Severe", "description": "Severe anxiety"}
                }
            },
            "stress": {
                "name": "Stress Assessment (PSS-10 Modified)",
                "description": "Measures perceived stress over the past month",
                "questions": [
                    "Been upset because of something that happened unexpectedly",
                    "Felt unable to control the important things in your life",
                    "Felt nervous and stressed",
                    "Felt confident about your ability to handle personal problems",
                    "Felt that things were going your way",
                    "Found that you could not cope with all the things you had to do",
                    "Been able to control irritations in your life",
                    "Felt that you were on top of things",
                    "Been angered because of things that happened outside of your control",
                    "Felt difficulties were piling up so high that you could not overcome them"
                ],
                "options": ["Never", "Almost never", "Sometimes", "Fairly often", "Very often"],
                "scores": [0, 1, 2, 3, 4],
                "reverse_scored": [3, 4, 6, 7],  # Questions that are reverse scored (0=4, 1=3, etc.)
                "interpretation": {
                    "0-13": {"level": "Low", "description": "Low perceived stress"},
                    "14-26": {"level": "Moderate", "description": "Moderate perceived stress"},
                    "27-40": {"level": "High", "description": "High perceived stress"}
                }
            },
            "wellbeing": {
                "name": "Well-being Assessment (WHO-5 Modified)",
                "description": "Measures current psychological well-being",
                "questions": [
                    "I have felt cheerful and in good spirits",
                    "I have felt calm and relaxed",
                    "I have felt active and vigorous",
                    "I woke up feeling fresh and rested",
                    "My daily life has been filled with things that interest me"
                ],
                "options": ["At no time", "Some of the time", "Less than half the time", 
                           "More than half the time", "Most of the time", "All the time"],
                "scores": [0, 1, 2, 3, 4, 5],
                "interpretation": {
                    "0-12": {"level": "Low", "description": "Low well-being, possible depression"},
                    "13-17": {"level": "Moderate", "description": "Moderate well-being"},
                    "18-25": {"level": "High", "description": "High well-being"}
                }
            }
        }
        
        # Initialize session state variables if they don't exist
        if "questionnaire_results" not in st.session_state:
            st.session_state.questionnaire_results = {}
        if "questionnaire_history" not in st.session_state:
            st.session_state.questionnaire_history = []
        if "current_questionnaire" not in st.session_state:
            st.session_state.current_questionnaire = None
        if "show_questionnaire" not in st.session_state:
            st.session_state.show_questionnaire = False
        if "assessment_completed" not in st.session_state:
            st.session_state.assessment_completed = False
        if "continue_to_conversation" not in st.session_state:
            st.session_state.continue_to_conversation = False
            
    def render_questionnaire_section(self):
        """Renders the questionnaire section in the Streamlit app"""
        
        st.subheader("Mental Health Assessment")
        
        # Check if user wants to continue to conversation
        if st.session_state.continue_to_conversation:
            # Update the EmotionAgent with the assessment results
            if st.session_state.questionnaire_history and "agent" in st.session_state:
                self.update_agent_with_results(st.session_state.questionnaire_history[-1])
                st.success("Assessment results added to your conversation context. Please go to the Conversation tab.")
                # Reset the flag
                st.session_state.continue_to_conversation = False
                st.session_state.assessment_completed = False
                return
        
        # If assessment was just completed, show results
        if st.session_state.assessment_completed:
            # FIX: Make sure to update the questionnaire_history before showing results
            if st.session_state.current_questionnaire or st.session_state.questionnaire_results:
                self.complete_questionnaire()
            self.show_assessment_results()
            return
        
        # Show questionnaire selection if not currently taking one
        if not st.session_state.show_questionnaire:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Quick Assessment (All Areas)", key="quick_assessment"):
                    st.session_state.current_questionnaire = "all"
                    st.session_state.show_questionnaire = True
                    st.rerun()
            
            with col2:
                selected = st.selectbox(
                    "Or select a specific assessment:",
                    ["Select an area to assess"] + list(self.questionnaires.keys()),
                    format_func=lambda x: "Select an area to assess" if x == "Select an area to assess" 
                              else self.questionnaires[x]["name"] if x in self.questionnaires 
                              else x
                )
                
                if selected != "Select an area to assess" and st.button("Start Assessment"):
                    st.session_state.current_questionnaire = selected
                    st.session_state.show_questionnaire = True
                    st.rerun()
            
            # Show previous results if available
            if st.session_state.questionnaire_history:
                self.show_assessment_history()
        
        # Show active questionnaire
        else:
            self.render_active_questionnaire()
    
    def render_active_questionnaire(self):
        """Renders the currently active questionnaire"""
        
        # Handle the "all" questionnaire type (shows all questionnaires in sequence)
        if st.session_state.current_questionnaire == "all":
            remaining = [q for q in self.questionnaires.keys() 
                        if q not in st.session_state.questionnaire_results]
            
            if not remaining:
                # All questionnaires completed, show results
                st.session_state.assessment_completed = True
                st.session_state.show_questionnaire = False
                st.rerun()
                return
                
            questionnaire_key = remaining[0]
        else:
            questionnaire_key = st.session_state.current_questionnaire
            
        questionnaire = self.questionnaires[questionnaire_key]
        
        # Display questionnaire header
        st.write(f"### {questionnaire['name']}")
        st.write(questionnaire['description'])
        
        # Create a form for the questionnaire
        with st.form(key=f"questionnaire_{questionnaire_key}"):
            responses = []
            
            for i, question in enumerate(questionnaire["questions"]):
                response = st.radio(
                    f"{i+1}. {question}",
                    options=questionnaire["options"],
                    key=f"q_{questionnaire_key}_{i}"
                )
                responses.append(response)
            
            submit = st.form_submit_button("Submit Responses")
            
            if submit:
                # Process and score the questionnaire
                score, interpretation = self.score_questionnaire(questionnaire_key, responses)
                
                # Store the results
                result = {
                    "questionnaire": questionnaire_key,
                    "name": questionnaire["name"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "score": score,
                    "interpretation": interpretation,
                    "responses": responses
                }
                
                st.session_state.questionnaire_results[questionnaire_key] = result
                
                # If this was part of the "all" assessment
                if st.session_state.current_questionnaire == "all":
                    remaining = [q for q in self.questionnaires.keys() 
                                if q not in st.session_state.questionnaire_results]
                    
                    if not remaining:
                        # All questionnaires completed
                        st.session_state.assessment_completed = True
                        st.session_state.show_questionnaire = False
                        st.rerun()
                    else:
                        # More questionnaires to go
                        st.rerun()
                else:
                    # Single questionnaire completed
                    st.session_state.assessment_completed = True
                    st.session_state.show_questionnaire = False
                    st.rerun()
    
    def score_questionnaire(self, questionnaire_key, responses):
        """Scores a completed questionnaire and returns the interpretation"""
        questionnaire = self.questionnaires[questionnaire_key]
        
        # Map responses to scores
        scores = []
        for i, response in enumerate(responses):
            option_index = questionnaire["options"].index(response)
            score_value = questionnaire["scores"][option_index]
            
            # Handle reverse scoring if applicable
            if "reverse_scored" in questionnaire and i in questionnaire["reverse_scored"]:
                score_value = questionnaire["scores"][-1] - score_value
                
            scores.append(score_value)
        
        total_score = sum(scores)
        
        # Find the correct interpretation range
        interpretation = None
        for score_range, interp_data in questionnaire["interpretation"].items():
            low, high = map(int, score_range.split("-"))
            if low <= total_score <= high:
                interpretation = interp_data
                break
        
        return total_score, interpretation
    
    def complete_questionnaire(self):
        """Handles questionnaire completion, showing results and updating history"""
        
        # Add results to history
        if st.session_state.current_questionnaire == "all":
            # For "all" questionnaire type, add all results as a set
            combined_result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "results": st.session_state.questionnaire_results.copy()
            }
            st.session_state.questionnaire_history.append(combined_result)
        else:
            # For single questionnaire, add just that result
            result = st.session_state.questionnaire_results[st.session_state.current_questionnaire]
            single_result = {
                "timestamp": result["timestamp"],
                "results": {st.session_state.current_questionnaire: result}
            }
            st.session_state.questionnaire_history.append(single_result)
        
        # Debug statement to confirm history was updated
        print(f"Assessment added to history. Total assessments: {len(st.session_state.questionnaire_history)}")
        
        # Reset the questionnaire state but keep assessment_completed as True
        st.session_state.show_questionnaire = False
        st.session_state.current_questionnaire = None
        st.session_state.questionnaire_results = {}
        
        # Don't reset assessment_completed here, we need it to show results

    def show_assessment_results(self):
        """Shows the results of the most recent assessment"""
        
        st.write("### Assessment Results")
        
        if not st.session_state.questionnaire_history:
            st.error("No assessment results found.")
            # DEBUG: Add information about what happened
            st.info("Debug info: The questionnaire was completed but results weren't saved to history.")
            return
        
        latest_assessment = st.session_state.questionnaire_history[-1]
        results = latest_assessment["results"]
        
        for area, result in results.items():
            questionnaire = self.questionnaires[area]
            interpretation = result["interpretation"]
            
            st.write(f"**{questionnaire['name']}**")
            st.write(f"Score: {result['score']} - {interpretation['level']}")
            st.write(f"Interpretation: {interpretation['description']}")
            st.write("---")
        
        # Use a session state variable instead of direct button action
        if st.button("Continue to Conversation", key="continue_btn"):
            st.session_state.continue_to_conversation = True
            st.rerun()
    
    def show_assessment_history(self):
        """Shows the history of past assessments"""
        
        with st.expander("View Previous Assessment Results"):
            if not st.session_state.questionnaire_history:
                st.write("No previous assessments available.")
                return
            
            # Allow user to select which assessment to view
            timestamps = [f"{h['timestamp']}" for h in st.session_state.questionnaire_history]
            selected_index = st.selectbox(
                "Select assessment:",
                range(len(timestamps)),
                format_func=lambda i: timestamps[i]
            )
            
            selected = st.session_state.questionnaire_history[selected_index]
            st.write(f"Assessment from: {selected['timestamp']}")
            
            # Display the selected assessment
            for area, result in selected["results"].items():
                questionnaire = self.questionnaires[area]
                interpretation = result["interpretation"]
                
                st.write(f"**{questionnaire['name']}**")
                st.write(f"Score: {result['score']} - {interpretation['level']}")
                st.write(f"Interpretation: {interpretation['description']}")
                st.write("---")
            
            # Button to use a previous assessment in conversation
            if st.button("Use This Assessment", key=f"use_assessment_{selected_index}"):
                if "agent" in st.session_state:
                    self.update_agent_with_results(selected)
                    st.success("Previous assessment loaded into conversation context.")
                else:
                    st.warning("Please complete emotion detection first in the Emotion Detection tab.")
    
    def update_agent_with_results(self, assessment):
        """Updates the EmotionAgent with questionnaire results"""
        
        if "agent" not in st.session_state:
            return
        
        # Extract key information from assessment results
        results_summary = {}
        overall_state = []
        
        for area, result in assessment["results"].items():
            results_summary[area] = {
                "score": result["score"],
                "level": result["interpretation"]["level"],
                "description": result["interpretation"]["description"]
            }
            
            # Add to overall state description based on significance
            if area == "mood" and result["interpretation"]["level"] in ["Moderate", "Moderately Severe", "Severe"]:
                overall_state.append(f"depressed ({result['interpretation']['level'].lower()})")
            
            if area == "anxiety" and result["interpretation"]["level"] in ["Moderate", "Severe"]:
                overall_state.append(f"anxious ({result['interpretation']['level'].lower()})")
            
            if area == "stress" and result["interpretation"]["level"] == "High":
                overall_state.append("highly stressed")
            
            if area == "wellbeing" and result["interpretation"]["level"] == "Low":
                overall_state.append("low wellbeing")
        
        # Format the overall state for the agent
        state_description = ", ".join(overall_state) if overall_state else "relatively stable"
        
        # Update the agent's context with assessment results
        assessment_context = (
            f"The user completed a psychological assessment on {assessment['timestamp']}. "
            f"Based on standardized questionnaires, they appear to be {state_description}. "
        )
        
        # Add specific details based on which assessments were completed
        if "mood" in results_summary:
            assessment_context += (
                f"Mood assessment indicates {results_summary['mood']['description']}. "
            )
        
        if "anxiety" in results_summary:
            assessment_context += (
                f"Anxiety assessment indicates {results_summary['anxiety']['description']}. "
            )
        
        if "stress" in results_summary:
            assessment_context += (
                f"Stress assessment indicates {results_summary['stress']['description']}. "
            )
        
        if "wellbeing" in results_summary:
            assessment_context += (
                f"Wellbeing assessment indicates {results_summary['wellbeing']['description']}. "
            )
        
        # Update agent system prompt with assessment information
        st.session_state.agent.add_assessment_results(assessment_context, results_summary)
        
        # Add a system message announcing the assessment results to the conversation
        st.session_state.agent.add_system_message(
            "Assessment completed. Results: " + assessment_context
        )