# -*- coding: utf-8 -*-
# @Time    : 2024/7/14 22:06
# @Author  : nongbin
# @FileName: llm.py
# @Software: PyCharm
# @Affiliation: tfswufe.edu.cn

import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field, create_model

# examples = [
#     {
#         "text": (
#             "Adam is a software engineer in Microsoft since 2009, "
#             "and last year he got an award as the Best Talent"
#         ),
#         "head": "Adam",
#         "head_type": "Person",
#         "relation": "WORKS_FOR",
#         "tail": "Microsoft",
#         "tail_type": "Company",
#     },
#     {
#         "text": (
#             "Adam is a software engineer in Microsoft since 2009, "
#             "and last year he got an award as the Best Talent"
#         ),
#         "head": "Adam",
#         "head_type": "Person",
#         "relation": "HAS_AWARD",
#         "tail": "Best Talent",
#         "tail_type": "Award",
#     },
#     {
#         "text": (
#             "Microsoft is a tech company that provide "
#             "several products such as Microsoft Word"
#         ),
#         "head": "Microsoft Word",
#         "head_type": "Product",
#         "relation": "PRODUCED_BY",
#         "tail": "Microsoft",
#         "tail_type": "Company",
#     },
#     {
#         "text": "Microsoft Word is a lightweight app that accessible offline",
#         "head": "Microsoft Word",
#         "head_type": "Product",
#         "relation": "HAS_CHARACTERISTIC",
#         "tail": "lightweight app",
#         "tail_type": "Characteristic",
#     },
#     {
#         "text": "Microsoft Word is a lightweight app that accessible offline",
#         "head": "Microsoft Word",
#         "head_type": "Product",
#         "relation": "HAS_CHARACTERISTIC",
#         "tail": "accessible offline",
#         "tail_type": "Characteristic",
#     },
# ]
examples = [
    {
        "text": (
            "张三是一名软件工程师，自2009年以来在华为工作，"
            "去年他获得了最佳人才奖。"
        ),
        "head": "张三",
        "head_type": "人",
        "relation": "工作于",
        "tail": "华为",
        "tail_type": "公司",
    },
    {
        "text": (
            "李四是一名数据科学家，目前在阿里巴巴工作，"
            "他在去年获得了最佳员工奖。"
        ),
        "head": "李四",
        "head_type": "人",
        "relation": "工作于",
        "tail": "阿里巴巴",
        "tail_type": "公司",
    },
    {
        "text": (
            "腾讯是一家科技公司，提供多种产品，例如微信和QQ。"
        ),
        "head": "微信",
        "head_type": "产品",
        "relation": "由...生产",
        "tail": "腾讯",
        "tail_type": "公司",
    },
    {
        "text": "微信是一款可以离线使用的轻量级应用。",
        "head": "微信",
        "head_type": "产品",
        "relation": "具有特征",
        "tail": "轻量级应用",
        "tail_type": "特征",
    },
    {
        "text": "QQ是一款社交应用，支持多种聊天功能。",
        "head": "QQ",
        "head_type": "产品",
        "relation": "具有特征",
        "tail": "多种聊天功能",
        "tail_type": "特征",
    },
]

# system_prompt = (
#     "# Knowledge Graph Instructions for GPT-4\n"
#     "## 1. Overview\n"
#     "You are a top-tier algorithm designed for extracting information in structured "
#     "formats to build a knowledge graph.\n"
#     "Try to capture as much information from the text as possible without "
#     "sacrifing accuracy. Do not add any information that is not explicitly "
#     "mentioned in the text\n"
#     "- **Nodes** represent entities and concepts.\n"
#     "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
#     "accessible for a vast audience.\n"
#     "## 2. Labeling Nodes\n"
#     "- **Consistency**: Ensure you use available types for node labels.\n"
#     "Ensure you use basic or elementary types for node labels.\n"
#     "- For example, when you identify an entity representing a person, "
#     "always label it as **'person'**. Avoid using more specific terms "
#     "like 'mathematician' or 'scientist'"
#     "  - **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
#     "names or human-readable identifiers found in the text.\n"
#     "- **Relationships** represent connections between entities or concepts.\n"
#     "Ensure consistency and generality in relationship types when constructing "
#     "knowledge graphs. Instead of using specific and momentary types "
#     "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
#     "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
#     "## 3. Coreference Resolution\n"
#     "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
#     "ensure consistency.\n"
#     'If an entity, such as "John Doe", is mentioned multiple times in the text '
#     'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
#     "always use the most complete identifier for that entity throughout the "
#     'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
#     "Remember, the knowledge graph should be coherent and easily understandable, "
#     "so maintaining consistency in entity references is crucial.\n"
#     "## 4. Strict Compliance\n"
#     "Adhere to the rules strictly. Non-compliance will result in termination."
# )
# system_prompt = (
#     "# 知识图谱指令\n"
#     "## 1. 概述\n"
#     "您是一个顶尖的算法，负责从提供的文本中提取结构化信息以构建知识图谱。\n"
#     "尽量从文本中捕获尽可能多的信息，而不牺牲准确性。不要添加任何文本中未明确提到的信息。\n"
#     "- **节点**表示实体和概念。\n"
#     "- 目标是实现知识图谱的简单性和清晰性，使其对广泛的受众可访问。\n"
#     "## 2. 节点标记\n"
#     "- **一致性**：确保使用可用类型作为节点标签。\n"
#     "确保使用基本或初步类型作为节点标签。\n"
#     "- 例如，当您识别一个代表人的实体时，始终将其标记为**'人'**。避免使用更具体的术语，如'数学家'或'科学家'。\n"
#     "  - **节点ID**：切勿使用整数作为节点ID。节点ID应为文本中找到的名称或人类可读的标识符。\n"
#     "- **关系**表示实体或概念之间的连接。\n"
#     "在构建知识图谱时，确保关系类型的一致性和普遍性。避免使用特定和瞬时的类型，如'成为教授'，而使用更一般和永恒的关系类型，如'教授'。确保使用一般和永恒的关系类型！\n"
#     "## 3. 共指解析\n"
#     "- **维护实体一致性**：提取实体时，确保一致性至关重要。\n"
#     '如果一个实体，如"约翰·多"，在文本中多次提到，但以不同的名称或代词（例如，"乔"，"他"）引用，则始终使用该实体的最完整标识符。\n'
#     "在这个例子中，使用\"约翰·多\"作为实体ID。\n"
#     "请记住，知识图谱应连贯且易于理解，因此维护实体引用的一致性至关重要。\n"
#     "## 4. 严格遵守\n"
#     "严格遵守规则。不遵守将导致终止。"
# )
system_prompt = (
    "您是一个文本抽取构建图谱专家，负责从提供的文本中提取尽可能多的信息去构建知识图谱。\n"
    "要求在确保准确性的同时，尽量从文本中捕获尽可能多的信息，构建出的知识图谱要结构清晰，图谱中的实体和关系数量要与文本长度成正比，做到每50个字能抽取出5个实体和5个关系，保证信息的丰富性。\n"
    "同时针对文本的内容，还需要做到：\n"
    "- 对于文本中表格内容的抽取，请特别注意解析表格中的标题和数据行，将每个单元格视为一个独立的信息点，并准确抽取表头与相应数据之间的关系。"
    "如果表格展示了具体的数值或成果，请将这些信息作为实体的属性或与其它实体间的具体关系进行记录，如果表格中的内容为“请文中抽取”，那就需要您通读全文，根据全文提到的当前实体信息根据表格标题进行填写"
    "- 对于文本中多次提及的同一实体，使用该实体最常见且完整的名称作为实体名称。如果实体在文本中以不同名称出现，请通过算法识别并统一这些变体到同一实体名称下，以确保知识图谱的一致性和完整性。"
)

default_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            # (
            #     "Tip: Make sure to answer in the correct format and do "
            #     "not include any explanations. "
            #     "Use the given format to extract information from the "
            #     "following input: {input}"
            # ),
            (
                "提示：确保以正确的格式回答并执行"
                "不包括任何解释"
                "使用给定的格式从中提取信息 "
                "以下输入: {input}"
            ),
        ),
    ]
)


def _get_additional_info(input_type: str) -> str:
    # Check if the input_type is one of the allowed values
    if input_type not in ["node", "relationship", "property"]:
        raise ValueError("input_type must be 'node', 'relationship', or 'property'")

    # Perform actions based on the input_type
    if input_type == "node":
        return (
            "Ensure you use basic or elementary types for node labels.\n"
            "For example, when you identify an entity representing a person, "
            "always label it as **'Person'**. Avoid using more specific terms "
            "like 'Mathematician' or 'Scientist'"
        )
    elif input_type == "relationship":
        return (
            "Instead of using specific and momentary types such as "
            "'BECAME_PROFESSOR', use more general and timeless relationship types like "
            "'PROFESSOR'. However, do not sacrifice any accuracy for generality"
        )
    elif input_type == "property":
        return ""
    return ""


def optional_enum_field(
    enum_values: Optional[List[str]] = None,
    description: str = "",
    input_type: str = "node",
    **field_kwargs: Any,
) -> Any:
    """Utility function to conditionally create a field with an enum constraint."""
    if enum_values:
        return Field(
            ...,
            enum=enum_values,
            description=f"{description}. Available options are {enum_values}",
            **field_kwargs,
        )
    else:
        additional_info = _get_additional_info(input_type)
        return Field(..., description=description + additional_info, **field_kwargs)


class _Graph(BaseModel):
    nodes: Optional[List]
    relationships: Optional[List]


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Person, Company, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted tail entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted tail entity like Person, Company, etc"
    )


def create_unstructured_prompt(
    node_labels: Optional[List[str]] = None, rel_types: Optional[List[str]] = None
) -> ChatPromptTemplate:
    node_labels_str = str(node_labels) if node_labels else ""
    rel_types_str = str(rel_types) if rel_types else ""
    # base_string_parts = [
    #     "You are a top-tier algorithm designed for extracting information in "
    #     "structured formats to build a knowledge graph. Your task is to identify "
    #     "the entities and relations requested with the user prompt from a given "
    #     "text. You must generate the output in a JSON format containing a list "
    #     'with JSON objects. Each object should have the keys: "head", '
    #     '"head_type", "relation", "tail", and "tail_type". The "head" '
    #     "key must contain the text of the extracted entity with one of the types "
    #     "from the provided list in the user prompt.",
    #     f'The "head_type" key must contain the type of the extracted head entity, '
    #     f"which must be one of the types from {node_labels_str}."
    #     if node_labels
    #     else "",
    #     f'The "relation" key must contain the type of relation between the "head" '
    #     f'and the "tail", which must be one of the relations from {rel_types_str}.'
    #     if rel_types
    #     else "",
    #     f'The "tail" key must represent the text of an extracted entity which is '
    #     f'the tail of the relation, and the "tail_type" key must contain the type '
    #     f"of the tail entity from {node_labels_str}."
    #     if node_labels
    #     else "",
    #     "Attempt to extract as many entities and relations as you can. Maintain "
    #     "Entity Consistency: When extracting entities, it's vital to ensure "
    #     'consistency. If an entity, such as "John Doe", is mentioned multiple '
    #     "times in the text but is referred to by different names or pronouns "
    #     '(e.g., "Joe", "he"), always use the most complete identifier for '
    #     "that entity. The knowledge graph should be coherent and easily "
    #     "understandable, so maintaining consistency in entity references is "
    #     "crucial.",
    #     "IMPORTANT NOTES:\n- Don't add any explanation and text.",
    # ]
    base_string_parts = [
        "你是一个顶尖的算法，负责从提供的文本中提取结构化信息以构建知识图谱。你的任务是根据用户提示从文本中识别实体和关系。"
        '请以JSON格式输出结果，格式为一个对象列表，每个对象包含以下键： "head_type", "relation", "tail", and "tail_type"'
        '"head"应包含从提供的列表中提取的实体文本及其类型，'
        f'head_type”键必须包含提取的头部实体的类型，必须是 {node_labels_str}中的内容 '
        if node_labels
        else "",
        f'The "relation" key must contain the type of relation between the "head" '
        f'and the "tail", which must be one of the relations from {rel_types_str}.'
        if rel_types
        else "",
        f'The "tail" key must represent the text of an extracted entity which is '
        f'the tail of the relation, and the "tail_type" key must contain the type '
        f"of the tail entity from {node_labels_str}."
        if node_labels
        else "",
        "尝试提取尽可能多的实体和关系。保持实体一致性：提取实体时，确保一致性至关重要。如果一个实体，如'约翰·多'，在文本中多次提到，但以不同的名称或代词（例如，'乔'，'他'）引用，则始终使用该实体的最完整标识符。知识图谱应连贯且易于理解，因此维护实体引用的一致性至关重要。",
        "重要提示：\n- 不要添加任何解释和额外文本。",
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))

    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

#     human_prompt = PromptTemplate(
#         template="""Based on the following example, extract entities and
# relations from the provided text.\n\n
# Use the following entity types, don't use other entity that is not defined below:
# # ENTITY TYPES:
# {node_labels}
#
# Use the following relation types, don't use other relation that is not defined below:
# # RELATION TYPES:
# {rel_types}
#
# Below are a number of examples of text and their extracted entities and relationships.
# {examples}
#
# For the following text, extract entities and relations as in the provided example.
# {format_instructions}\nText: {input}""",
#         input_variables=["input"],
#         partial_variables={
#             "format_instructions": parser.get_format_instructions(),
#             "node_labels": node_labels,
#             "rel_types": rel_types,
#             "examples": examples,
#         },
#     )
    human_prompt = PromptTemplate(
        template="""根据以下示例，从提供的文本中提取实体和关系。\n\n
            请使用以下实体类型，不要使用未在下面定义的其他实体：
            # 实体类型：
            {node_labels}

            请使用以下关系类型，不要使用未在下面定义的其他关系：
            # 关系类型：
            {rel_types}

            以下是一些文本示例及其提取的实体和关系：
            {examples}

            对于以下文本，请像提供的示例一样提取实体和关系。
    {format_instructions}\nText: {input}""",
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": node_labels,
            "rel_types": rel_types,
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )
    return chat_prompt


def create_simple_model(
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[List[str]] = None,
    node_properties: Union[bool, List[str]] = False,
) -> Type[_Graph]:
    """
    Simple model allows to limit node and/or relationship types.
    Doesn't have any node or relationship properties.
    """

    node_fields: Dict[str, Tuple[Any, Any]] = {
        "id": (
            str,
            Field(..., description="Name or human-readable unique identifier."),
        ),
        "type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the node.",
                input_type="node",
            ),
        ),
    }
    if node_properties:
        if isinstance(node_properties, list) and "id" in node_properties:
            raise ValueError("The node property 'id' is reserved and cannot be used.")
        # Map True to empty array
        node_properties_mapped: List[str] = (
            [] if node_properties is True else node_properties
        )

        class Property(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                node_properties_mapped,
                description="Property key.",
                input_type="property",
            )
            value: str = Field(..., description="value")

        node_fields["properties"] = (
            Optional[List[Property]],
            Field(None, description="List of node properties"),
        )
    SimpleNode = create_model("SimpleNode", **node_fields)  # type: ignore

    class SimpleRelationship(BaseModel):
        """Represents a directed relationship between two nodes in a graph."""

        source_node_id: str = Field(
            description="Name or human-readable unique identifier of source node"
        )
        source_node_type: str = optional_enum_field(
            node_labels,
            description="The type or label of the source node.",
            input_type="node",
        )
        target_node_id: str = Field(
            description="Name or human-readable unique identifier of target node"
        )
        target_node_type: str = optional_enum_field(
            node_labels,
            description="The type or label of the target node.",
            input_type="node",
        )
        type: str = optional_enum_field(
            rel_types,
            description="The type of the relationship.",
            input_type="relationship",
        )

    class DynamicGraph(_Graph):
        """Represents a graph document consisting of nodes and relationships."""

        nodes: Optional[List[SimpleNode]] = Field(description="List of nodes")  # type: ignore
        relationships: Optional[List[SimpleRelationship]] = Field(
            description="List of relationships"
        )

    return DynamicGraph


def map_to_base_node(node: Any) -> Node:
    """Map the SimpleNode to the base Node."""
    properties = {}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value
    return Node(id=node.id, type=node.type, properties=properties)


def map_to_base_relationship(rel: Any) -> Relationship:
    """Map the SimpleRelationship to the base Relationship."""
    source = Node(id=rel.source_node_id, type=rel.source_node_type)
    target = Node(id=rel.target_node_id, type=rel.target_node_type)
    return Relationship(source=source, target=target, type=rel.type)


def _parse_and_clean_json(
    argument_json: Dict[str, Any],
) -> Tuple[List[Node], List[Relationship]]:
    if 'Items' in argument_json["nodes"]:
        argument_json["nodes"] = argument_json['nodes']['Items']
    if 'Items' in argument_json["relationships"]:
        argument_json["relationships"] = argument_json["relationships"]['Items']
    nodes = []
    for node in argument_json["nodes"]:
        if not node.get("id"):  # Id is mandatory, skip this node
            continue
        nodes.append(
            Node(
                id=node["id"],
                type=node.get("type"),
            )
        )
    relationships = []
    for rel in argument_json["relationships"]:
        # Mandatory props
        if (
            not rel.get("source_node_id")
            or not rel.get("target_node_id")
            or not rel.get("type")
        ):
            continue

        # Node type copying if needed from node list
        if not rel.get("source_node_type"):
            try:
                rel["source_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["source_node_id"]
                ][0]
            except IndexError:
                rel["source_node_type"] = None
        if not rel.get("target_node_type"):
            try:
                rel["target_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["target_node_id"]
                ][0]
            except IndexError:
                rel["target_node_type"] = None

        source_node = Node(
            id=rel["source_node_id"],
            type=rel["source_node_type"],
        )
        target_node = Node(
            id=rel["target_node_id"],
            type=rel["target_node_type"],
        )
        relationships.append(
            Relationship(
                source=source_node,
                target=target_node,
                type=rel["type"],
            )
        )
    return nodes, relationships


def _format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id=el.id.title() if isinstance(el.id, str) else el.id,
            type=el.type.capitalize(),
            properties=el.properties,
        )
        for el in nodes
    ]


def _format_relationships(rels: List[Relationship]) -> List[Relationship]:
    return [
        Relationship(
            source=_format_nodes([el.source])[0],
            target=_format_nodes([el.target])[0],
            type=el.type.replace(" ", "_").upper(),
        )
        for el in rels
    ]


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def _convert_to_graph_document(
    raw_schema: Dict[Any, Any],
) -> Tuple[List[Node], List[Relationship]]:
    # If there are validation errors
    if not raw_schema["parsed"]:
        try:
            try:  # OpenAI type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["tool_calls"][0]["function"][
                        "arguments"
                    ]
                )
            except Exception:  # Google type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["function_call"]["arguments"]
                )

            nodes, relationships = _parse_and_clean_json(argument_json)
        except Exception:  # If we can't parse JSON
            return ([], [])
    else:  # If there are no validation errors use parsed pydantic object
        parsed_schema: _Graph = raw_schema["parsed"]
        nodes = (
            [map_to_base_node(node) for node in parsed_schema.nodes]
            if parsed_schema.nodes
            else []
        )

        relationships = (
            [map_to_base_relationship(rel) for rel in parsed_schema.relationships]
            if parsed_schema.relationships
            else []
        )
    # Title / Capitalize
    return _format_nodes(nodes), _format_relationships(relationships)


class LLMGraphTransformer:
    """Transform documents into graph-based documents using a LLM.

    It allows specifying constraints on the types of nodes and relationships to include
    in the output graph. The class doesn't support neither extract and node or
    relationship properties

    Args:
        llm (BaseLanguageModel): An instance of a language model supporting structured
          output.
        allowed_nodes (List[str], optional): Specifies which node types are
          allowed in the graph. Defaults to an empty list, allowing all node types.
        allowed_relationships (List[str], optional): Specifies which relationship types
          are allowed in the graph. Defaults to an empty list, allowing all relationship
          types.
        prompt (Optional[ChatPromptTemplate], optional): The prompt to pass to
          the LLM with additional instructions.
        strict_mode (bool, optional): Determines whether the transformer should apply
          filtering to strictly adhere to `allowed_nodes` and `allowed_relationships`.
          Defaults to True.

    Example:
        .. code-block:: python
            from langchain_experimental.graph_transformers import LLMGraphTransformer
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI

            llm=ChatOpenAI(temperature=0)
            transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=["Person", "Organization"])

            doc = Document(page_content="Elon Musk is suing OpenAI")
            graph_documents = transformer.convert_to_graph_documents([doc])
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = [],
        allowed_relationships: List[str] = [],
        prompt: Optional[ChatPromptTemplate] = None,
        strict_mode: bool = True,
        node_properties: Union[bool, List[str]] = False,
        use_function_call: bool = True
    ) -> None:
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self._function_call = use_function_call
        # Check if the LLM really supports structured output
        try:
            llm.with_structured_output(_Graph)
        except NotImplementedError:
            self._function_call = False
        if not self._function_call:
            if node_properties:
                raise ValueError(
                    "The 'node_properties' parameter cannot be used "
                    "in combination with a LLM that doesn't support "
                    "native function calling."
                )
            try:
                import json_repair

                self.json_repair = json_repair
            except ImportError:
                raise ImportError(
                    "Could not import json_repair python package. "
                    "Please install it with `pip install json-repair`."
                )
            prompt = prompt or create_unstructured_prompt(
                allowed_nodes, allowed_relationships
            )
            self.chain = prompt | llm
        else:
            # Define chain
            schema = create_simple_model(
                allowed_nodes, allowed_relationships, node_properties
            )
            structured_llm = llm.with_structured_output(schema, include_raw=True)
            prompt = prompt or default_prompt
            self.chain = prompt | structured_llm

    def process_response(self, document: Document) -> GraphDocument:
        """
        Processes a single document, transforming it into a graph document using
        an LLM based on the model's schema and constraints.
        """
        text = document.page_content
        print("开始抽取")
        # 抽取关键
        raw_schema = self.chain.invoke({"input": text})
        if self._function_call:
            raw_schema = cast(Dict[Any, Any], raw_schema)
            nodes, relationships = _convert_to_graph_document(raw_schema)
        else:
            nodes_set = set()
            relationships = []
            parsed_json = self.json_repair.loads(raw_schema.content)
            for rel in parsed_json:
                # # Nodes need to be deduplicated using a set
                # print(f"rel: {rel}, type: {type(rel)}")
                # nodes_set.add((rel["head"], rel["head_type"]))
                # nodes_set.add((rel["tail"], rel["tail_type"]))
                #
                # source_node = Node(id=rel["head"], type=rel["head_type"])
                # target_node = Node(id=rel["tail"], type=rel["tail_type"])
                # relationships.append(
                #     Relationship(
                #         source=source_node, target=target_node, type=rel["relation"]
                #     )
                # )
                if all(key in rel for key in ["head", "head_type", "tail", "tail_type", "relation"]):
                    print(f"rel: {rel}, type: {type(rel)}")
                    nodes_set.add((rel["head"], rel["head_type"]))
                    nodes_set.add((rel["tail"], rel["tail_type"]))

                    source_node = Node(id=rel["head"], type=rel["head_type"])
                    target_node = Node(id=rel["tail"], type=rel["tail_type"])
                    relationships.append(
                        Relationship(
                            source=source_node, target=target_node, type=rel["relation"]
                        )
                    )
                else:
                    print(f"跳过缺少字段的关系: {rel}")  # 可以选择记录或处理缺少字段的情况
            # Create nodes list
            nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]

        # Strict mode filtering
        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type.lower()
                    in [el.lower() for el in self.allowed_relationships]
                ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    def convert_to_graph_documents(
        self, documents: Sequence[Document]
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            **kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        return [self.process_response(document) for document in documents]

    async def aprocess_response(self, document: Document) -> GraphDocument:
        """
        Asynchronously processes a single document, transforming it into a
        graph document.
        """
        text = document.page_content
        raw_schema = await self.chain.ainvoke({"input": text})
        raw_schema = cast(Dict[Any, Any], raw_schema)
        nodes, relationships = _convert_to_graph_document(raw_schema)

        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type.lower()
                    in [el.lower() for el in self.allowed_relationships]
                ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    async def aconvert_to_graph_documents(
        self, documents: Sequence[Document]
    ) -> List[GraphDocument]:
        """
        Asynchronously convert a sequence of documents into graph documents.
        """
        tasks = [
            asyncio.create_task(self.aprocess_response(document))
            for document in documents
        ]
        results = await asyncio.gather(*tasks)
        return results
