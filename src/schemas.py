from pydantic import BaseModel,Field
from typing import Literal,List

class QueryClassification(BaseModel):
    
    complexity: Literal["simple", "complex"] = Field(
        description="Classifies the given user query as either 'simple' or 'complex' based on its complexity."
    )

class QueryDecompose(BaseModel):

    sub_queries : List = Field(
        description= "A list of sub-queries generated from the complex query"
    )

class GradeDocuments(BaseModel):

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class ResponseGenerator(BaseModel):

    output : str = Field(
        description="Response to the Provided Query"
    )

