# EmotionAgent.py

class EmotionAgent:
    def __init__(self, llm_model, face_emotion="neutral", speech_emotion="neutral"):
        self.model = llm_model
        self.face_emotion = face_emotion
        self.speech_emotion = speech_emotion
        self.final_emotion = self.decide_emotion()
        
        # New assessment-related attributes
        self.assessment_results = None
        self.assessment_summary = None
        
        self.system_prompt = self.build_system_prompt()
        self.full_context = [{"role": "system", "content": self.system_prompt}]
        self.messages = []

    def decide_emotion(self):
        """Decides which emotion to use based on face and speech emotions."""
        # If both emotions are close (same or similar), use it
        if self.face_emotion == self.speech_emotion:
            return self.face_emotion
        
        # If one is neutral and the other is not, prefer the non-neutral
        if self.face_emotion == "neutral" and self.speech_emotion != "neutral":
            return self.speech_emotion
        if self.speech_emotion == "neutral" and self.face_emotion != "neutral":
            return self.face_emotion
        
        # If emotions are very different, ignore and return neutral
        very_different_pairs = [
            ("happy", "angry"), ("angry", "happy"),
            ("happy", "sad"), ("sad", "happy"),
            ("angry", "surprise"), ("surprise", "angry"),
            ("sad", "surprise"), ("surprise", "sad"),
            ("fear", "happy"), ("happy", "fear")
        ]
        if (self.face_emotion, self.speech_emotion) in very_different_pairs or \
           (self.speech_emotion, self.face_emotion) in very_different_pairs:
            return "neutral"
        
        # Default: fallback to face emotion
        return self.face_emotion

    def build_system_prompt(self):
        """Creates the system prompt considering the final emotion and assessment results."""
        base = (
            "You are a professional therapist specializing in mental health. "
            "You listen with empathy, validate emotions, and offer guidance without judgment. "
            "You respond in a kind, clear, and emotionally adaptive manner focused on emotional well-being."
        )
        
        # Add emotion context
        if self.final_emotion and self.final_emotion.lower() != "neutral":
            base += f" The user appears to be feeling {self.final_emotion.lower()}, so respond appropriately for someone who feels this way."
        else:
            base += " The user's emotion appears to be neutral; ask gentle, open-ended questions to explore how they feel."
        
        # Add assessment results context if available
        if self.assessment_results:
            base += f" {self.assessment_results}"
            
            # Add therapeutic guidance based on assessment results
            base += self._get_therapeutic_guidance()
        
        return base
    
    def _get_therapeutic_guidance(self):
        """Generates therapeutic guidance based on assessment results."""
        if not self.assessment_summary:
            return ""
        
        guidance = " As a therapist, you should:"
        
        # Add specific guidance based on mood assessment
        if "mood" in self.assessment_summary:
            mood_level = self.assessment_summary["mood"]["level"]
            if mood_level in ["Moderate", "Moderately Severe", "Severe"]:
                guidance += (
                    " Validate feelings of depression without reinforcing negative thought patterns."
                    " Use gentle encouragement and focus on small achievable goals."
                    " Introduce concepts of behavioral activation."
                )
            elif mood_level == "Mild":
                guidance += (
                    " Acknowledge their feelings while highlighting their strengths."
                    " Encourage activities that have previously brought them joy."
                )
        
        # Add specific guidance based on anxiety assessment
        if "anxiety" in self.assessment_summary:
            anxiety_level = self.assessment_summary["anxiety"]["level"]
            if anxiety_level in ["Moderate", "Severe"]:
                guidance += (
                    " Help ground the user when discussing anxiety-provoking topics."
                    " Introduce simple breathing techniques when appropriate."
                    " Validate their concerns while gently challenging catastrophic thinking."
                )
            elif anxiety_level == "Mild":
                guidance += (
                    " Acknowledge their anxiety while helping them examine thought patterns."
                    " Suggest mindfulness practices that might help them stay present."
                )
        
        # Add specific guidance based on stress assessment
        if "stress" in self.assessment_summary:
            stress_level = self.assessment_summary["stress"]["level"]
            if stress_level == "High":
                guidance += (
                    " Help them identify specific stressors and explore coping strategies."
                    " Discuss boundaries and self-care practices."
                    " Validate that their stress response is normal given their circumstances."
                )
        
        # Add specific guidance based on wellbeing assessment
        if "wellbeing" in self.assessment_summary:
            wellbeing_level = self.assessment_summary["wellbeing"]["level"]
            if wellbeing_level == "Low":
                guidance += (
                    " Focus on identifying small sources of meaning and purpose."
                    " Explore what has previously contributed to their sense of wellbeing."
                )
            elif wellbeing_level == "High":
                guidance += (
                    " Acknowledge their positive state while exploring how to maintain it."
                    " Use their current strengths as resources for addressing any challenges."
                )
        
        return guidance

    def update_emotions(self, face_emotion, speech_emotion):
        """Updates emotions and system prompt."""
        self.face_emotion = face_emotion
        self.speech_emotion = speech_emotion
        self.final_emotion = self.decide_emotion()
        self.system_prompt = self.build_system_prompt()
        # Update conversation history with updated system prompt
        self.full_context = [{"role": "system", "content": self.system_prompt}] + self.messages

    def add_assessment_results(self, assessment_context, results_summary):
        """Adds psychological assessment results to the agent's context."""
        self.assessment_results = assessment_context
        self.assessment_summary = results_summary
        # Rebuild the system prompt with the new assessment information
        self.system_prompt = self.build_system_prompt()
        # Update conversation context with new system prompt
        self.full_context = [{"role": "system", "content": self.system_prompt}] + self.messages

    def add_user_input(self, user_input):
        """Adds user message to conversation history."""
        self.messages.append({"role": "user", "content": user_input})
        self.full_context.append({"role": "user", "content": user_input})

    def add_system_message(self, system_message):
        """Adds a system message to the conversation (visible to user)."""
        self.messages.append({"role": "system", "content": system_message})
        # We don't add this to full_context as it's already reflected in the system prompt

    def get_response(self):
        """Generates response from the model."""
        prompt_text = ''
        for msg in self.full_context:
            if msg['role'] == 'system':
                prompt_text += msg['content'] + "\n"
            elif msg['role'] == 'user':
                prompt_text += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt_text += f"Therapist: {msg['content']}\n"
        prompt_text += "Therapist:"
        
        response = self.model.respond(prompt_text)
        # Store assistant response
        self.messages.append({"role": "assistant", "content": response})
        self.full_context.append({"role": "assistant", "content": response})
        return response
    
    def get_therapeutic_insights(self):
        """
        Generates insights about the user's mental state based on 
        emotional and assessment data collected.
        """
        if not self.messages or not (self.final_emotion != "neutral" or self.assessment_results):
            return None
        
        # Create a prompt specifically for generating insights
        insight_prompt = (
            "Based on the conversation history and assessment data, "
            "provide a brief therapeutic insight about the user's current mental state. "
            "Focus on patterns, strengths, and areas that might benefit from attention. "
            "Keep it concise (2-3 sentences)."
        )
        
        # Add context from emotions and assessments
        if self.final_emotion != "neutral":
            insight_prompt += f" The user has displayed {self.final_emotion} emotions."
        
        if self.assessment_results:
            insight_prompt += f" {self.assessment_results}"
        
        # Add the last few messages for context
        recent_messages = self.messages[-min(5, len(self.messages)):]
        message_context = ""
        for msg in recent_messages:
            if msg['role'] == 'user':
                message_context += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                message_context += f"Therapist: {msg['content']}\n"
        
        insight_prompt += f"\nRecent conversation:\n{message_context}\n\nTherapeutic insight:"
        
        # Generate the insight using the LLM
        insight = self.model.respond(insight_prompt)
        return insight