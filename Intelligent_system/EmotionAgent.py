# --- Updated EmotionAgent Definition ---

class EmotionAgent:
    def __init__(self, llm_model, face_emotion="neutral", speech_emotion="neutral"):
        self.model = llm_model
        self.face_emotion = face_emotion
        self.speech_emotion = speech_emotion
        self.final_emotion = self.decide_emotion()
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
        """Creates the system prompt considering the final emotion."""
        base = (
            "You are a professional therapist specializing in mental health. "
            "You listen with empathy, validate emotions, and offer guidance without judgment. "
            "You respond in a kind, clear, and emotionally adaptive manner focused on emotional well-being."
        )
        if self.final_emotion and self.final_emotion.lower() != "neutral":
            base += f" The user appears to be feeling {self.final_emotion.lower()}, so respond appropriately for someone who feels this way."
        else:
            base += " The user's emotion appears to be neutral; ask gentle, open-ended questions to explore how they feel."
        return base

    def update_emotions(self, face_emotion, speech_emotion):
        """Updates emotions and system prompt."""
        self.face_emotion = face_emotion
        self.speech_emotion = speech_emotion
        self.final_emotion = self.decide_emotion()
        self.system_prompt = self.build_system_prompt()
        # Reset conversation history with updated system prompt
        self.full_context = [{"role": "system", "content": self.system_prompt}] + self.messages

    def add_user_input(self, user_input):
        """Adds user message to conversation history."""
        self.messages.append({"role": "user", "content": user_input})
        self.full_context.append({"role": "user", "content": user_input})

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
