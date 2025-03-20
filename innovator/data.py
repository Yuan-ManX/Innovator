from typing import List, Optional
from docarray import BaseDoc, DocList


class Info(BaseDoc):
    """A structured information document used for agent communication."""
    
    content: List[str] = []  # Ensures content is a list of strings
    instruction: str = ''
    agent_id: str = ''  # The agent's profile identifier
    role: str = 'user'  # Can be: system / user / assistant
    cause_by: Optional[str] = None  # Optional trigger event

    @property
    def Info_str(self) -> str:
        """Returns a structured string representation of the Info object."""
        return f"[{self.role} - {self.agent_id}]: {', '.join(self.content)}"

    def __repr__(self) -> str:
        """Readable string representation of the Info object."""
        return f"Info(role={self.role}, agent_id={self.agent_id}, content={self.content})"

    def to_dict(self) -> dict:
        """Converts the object into a dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
            "agent_id": self.agent_id,
            "cause_by": self.cause_by
        }


class ResponseListDoc(BaseDoc):
    """A response document containing a list of responses."""
    
    response_list: List[str] = []  # List of responses


class Response(BaseDoc):
    """A structured response document that supports multiple modalities."""
    
    image: DocList[ResponseListDoc] = DocList.empty(ResponseListDoc)
    text: DocList[ResponseListDoc] = DocList.empty(ResponseListDoc)
    audio: DocList[ResponseListDoc] = DocList.empty(ResponseListDoc)
    video: DocList[ResponseListDoc] = DocList.empty(ResponseListDoc)

    def add_text_response(self, response: str):
        """Adds a text response to the response list."""
        if not self.text:
            self.text = DocList([ResponseListDoc(response_list=[response])])
        else:
            self.text[0].response_list.append(response)

    def add_image_response(self, image_data: str):
        """Adds an image response to the response list."""
        if not self.image:
            self.image = DocList([ResponseListDoc(response_list=[image_data])])
        else:
            self.image[0].response_list.append(image_data)

    @classmethod
    def from_text(cls, text_response: List[str]) -> "Response":
        """Creates a Response object from text responses."""
        return cls(text=DocList([ResponseListDoc(response_list=text_response)]))

    @classmethod
    def from_image(cls, image_response: List[str]) -> "Response":
        """Creates a Response object from image responses."""
        return cls(image=DocList([ResponseListDoc(response_list=image_response)]))

    def to_dict(self) -> dict:
        """Converts the response object into a dictionary format."""
        return {
            "text": [r.response_list for r in self.text],
            "image": [r.response_list for r in self.image],
            "audio": [r.response_list for r in self.audio],
            "video": [r.response_list for r in self.video],
        }


info = Info(content=["Hello", "How are you?"], agent_id="agent_001", role="user", cause_by="greeting")
print(info.Info_str)  
# 输出: [user - agent_001]: Hello, How are you?

response = Response.from_text(["Hello!", "I'm good."])
response.add_text_response("What about you?")
print(response.to_dict())
# 输出:
# {
#     "text": [["Hello!", "I'm good.", "What about you?"]],
#     "image": [],
#     "audio": [],
#     "video": []
# }
