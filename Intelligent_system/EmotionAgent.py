# --- EmotionAgent Definition ---
class EmotionAgent:
    def __init__(self, llm_model, emotion="neutral"):
        self.model = llm_model
        self.emotion = emotion
        self.system_prompt = self.build_system_prompt()
        self.full_context = [{"role": "system", "content": self.system_prompt}]
        self.messages = []

    def build_system_prompt(self):
        base = (
            "You are a professional therapist specializing in mental health. "
            "You listen with empathy, validate emotions, and offer guidance without judgment. "
            "You respond in a kind, clear, and emotionally adaptive manner focused on emotional well-being."
        )
        if self.emotion and self.emotion.lower() != "neutral":
            base += f" The user appears to be feeling {self.emotion.lower()}, so respond appropriately for someone who feels this way."
        else:
            base += " The user's emotion appears to be neutral; ask gentle, open-ended questions to explore how they feel."
        return base

    def update_emotion(self, new_emotion):
        self.emotion = new_emotion
        self.system_prompt = self.build_system_prompt()
        # reset context with updated system prompt
        self.full_context = [{"role": "system", "content": self.system_prompt}] + self.messages

    def add_user_input(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        self.full_context.append({"role": "user", "content": user_input})

    def get_response(self):
        # build the complete prompt text
        prompt_text = ''
        for msg in self.full_context:
            if msg['role'] == 'system':
                prompt_text += msg['content'] + "\n"
            elif msg['role'] == 'user':
                prompt_text += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt_text += f"Therapist: {msg['content']}\n"
        prompt_text += "Therapist:"
        # get model response
        response = self.model.respond(prompt_text)
        # store assistant response
        self.messages.append({"role": "assistant", "content": response})
        self.full_context.append({"role": "assistant", "content": response})
        return response