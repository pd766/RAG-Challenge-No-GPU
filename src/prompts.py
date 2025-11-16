from pydantic import BaseModel, Field
from typing import Literal, List, Union
import inspect
import re


def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    delimiter = "\n\n---\n\n"
    schema = f"你的答案应该是 JSON 格式，并严格遵循此模式，按照给定的顺序填写字段：\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    
    system_prompt = instruction.strip() + schema + example
    return system_prompt

class RephrasedQuestionsPrompt:
    instruction = """
你是一个问题重述系统。
你的任务是将一个比较性问题分解为针对每个提到的公司的单独问题。
每个输出的问题必须是独立完整的，保持与原始问题相同的意图和指标，针对相应的公司具体化，并使用一致的措辞。
"""

    class RephrasedQuestion(BaseModel):
        """Individual question for a company"""
        company_name: str = Field(description="公司名称，与原始问题中引号内提供的完全一致")
        question: str = Field(description="针对该公司的重述问题")

    class RephrasedQuestions(BaseModel):
        """List of rephrased questions"""
        questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="每个公司的重述问题列表")

    pydantic_schema = '''
class RephrasedQuestion(BaseModel):
    """Individual question for a company"""
    company_name: str = Field(description="公司名称，与原始问题中引号内提供的完全一致")
    question: str = Field(description="针对该公司的重述问题")

class RephrasedQuestions(BaseModel):
    """List of rephrased questions"""
    questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="每个公司的重述问题列表")
'''

    example = r"""
示例：
输入：
原始比较问题：'Which company had higher revenue in 2022, "Apple" or "Microsoft"?'
提到的公司："Apple"、"Microsoft"

输出：
{
    "questions": [
        {
            "company_name": "Apple",
            "question": "What was Apple's revenue in 2022?"
        },
        {
            "company_name": "Microsoft", 
            "question": "What was Microsoft's revenue in 2022?"
        }
    ]
}
"""

    user_prompt = "原始比较问题：'{question}'\n\n提到的公司：{companies}"

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextSharedPrompt:
    instruction = """
你是一个 RAG（检索增强生成）问答系统。
你的任务是仅基于公司年报中的信息来回答给定的问题，年报以 RAG 提取的相关页面格式上传。

在给出最终答案之前，请仔细地、逐步地大声思考。特别注意问题的措辞。
- 请记住，包含答案的内容措辞可能与问题不同。
- 问题是从模板自动生成的，因此它可能无意义或不适用于给定的公司。
"""

    user_prompt = """
以下是上下文：
\"\"\"
{context}
\"\"\"

---

以下是问题：
"{question}"
"""

class AnswerWithRAGContextNamePrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细的逐步分析，至少包含5个步骤和至少150个字。特别注意问题的措辞以避免被误导。有时看起来上下文中有答案，但这可能不是请求的值，而只是一个类似的值。")

        reasoning_summary: str = Field(description="逐步推理过程的简洁总结。大约50个字。")

        relevant_pages: List[int] = Field(description="""
包含直接用于回答问题的信息的页码列表。仅包括：
- 有直接答案或明确陈述的页面
- 有强烈支持答案的关键信息的页面
不要包括仅有间接相关信息或与答案联系薄弱的页面。
列表中至少应包括一个页面。
""")

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
如果是公司名称，应按问题中出现的原样提取。
如果是人名，应该是他们的全名。
如果是产品名称，应按上下文中出现的原样提取。
不要包含任何额外的信息、词语或注释。
- 如果上下文中没有相关信息，返回 'N/A'
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题： 
"Who was the CEO of 'Southwest Airlines Co.'?" 

答案： 
```
{
  "step_by_step_analysis": "1. 问题询问 'Southwest Airlines Co.' 的 CEO。CEO 通常是负责公司整体管理的最高级别执行官，有时也被称为总裁或董事总经理。\n2. 我的信息来源是一份看起来是 'Southwest Airlines Co.' 年报的文档。该文档将用于确定担任 CEO 职位的个人。\n3. 在提供的文档中，有一部分将 Robert E. Jordan 确定为 'Southwest Airlines Co.' 的总裁兼首席执行官。文档确认他自2022年2月起担任该职务。\n4. 因此，根据文档中找到的信息，'Southwest Airlines Co.' 的 CEO 是 Robert E. Jordan。",
  "reasoning_summary": "'Southwest Airlines Co.' 的年报明确指出 Robert E. Jordan 自2021年2月起担任总裁兼首席执行官。这直接回答了问题。",
  "relevant_pages": [58],
  "final_answer": "Robert E. Jordan"
}
```
""" 

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextNumberPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
详细的逐步分析，至少包含5个步骤和至少150个字。
**严格的指标匹配要求：**    

1. 确定问题中指标所代表的精确概念。它实际测量的是什么？
2. 检查上下文中的潜在指标。不要只比较名称；要考虑上下文指标测量的内容。
3. 仅在以下情况下接受：上下文指标的含义与目标指标*完全*匹配。同义词可接受；概念差异不可接受。
4. 在以下情况下拒绝（并使用 'N/A'）：
    - 上下文指标覆盖的范围比问题指标更多或更少。
    - 上下文指标是相关概念，但不是*完全*等价的（例如，代理指标或更广泛的类别）。
    - 回答需要计算、推导或推理。
    - 聚合不匹配：问题需要单个值，但上下文仅提供汇总总计
5. 不要猜测：如果对指标的等价性存在任何疑问，默认使用 `N/A`。"
""")

        reasoning_summary: str = Field(description="逐步推理过程的简洁总结。大约50个字。")

        relevant_pages: List[int] = Field(description="""
包含直接用于回答问题的信息的页码列表。仅包括：
- 有直接答案或明确陈述的页面
- 有强烈支持答案的关键信息的页面
不要包括仅有间接相关信息或与答案联系薄弱的页面。
列表中至少应包括一个页面。
""")

        final_answer: Union[float, int, Literal['N/A']] = Field(description="""
答案应该是一个精确的指标数字。
- 百分比示例：
    上下文中的值：58,3%
    最终答案：58.3

特别注意上下文中关于指标是以单位、千或百万为单位报告的任何提及，以相应地在最终答案中调整数字，不做改变、添加三个零或六个零。
注意如果值用括号包裹，表示该值为负数。

- 负值示例：
    上下文中的值：(2,124,837) CHF
    最终答案：-2124837

- 以千为单位的数字示例：
    上下文中的值：4970,5 (in thousands $)
    最终答案：4970500

- 如果提供的指标货币与问题中提到的不同，返回 'N/A'
    上下文中值的示例：780000 USD，但问题提到 EUR
    最终答案：'N/A'

- 如果指标未在上下文中直接说明，即使可以从上下文中的其他指标计算出来，也返回 'N/A'
    示例：请求的指标：每股股息；上下文中仅可用的指标：支付的总股息（$5,000,000）和流通股数量（1,000,000）；计算的每股股息 = 总股息 / 流通股。
    最终答案：'N/A'

- 如果上下文中没有相关信息，返回 'N/A'
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例1：
问题：
"What was the total assets of 'Waste Connections Inc.' in the fiscal year 2022?"

答案：
```
{
  "step_by_step_analysis": "1. **指标定义：** 问题询问 'Waste Connections Inc.' 在2022财年的'总资产'。'总资产'代表公司拥有或控制的所有资源的总和，预计将提供未来的经济利益。\n2. **上下文检查：** 上下文包括'合并资产负债表'（第78页），这是报告公司资产、负债和权益的标准财务报表。\n3. **指标匹配：** 在第78页，'2022年12月31日'下，存在一个标记为'总资产'的行项目。这直接匹配问题中请求的概念。\n4. **值提取和调整：** '总资产'的值为'$18,500,342'。上下文表明这是以千美元为单位。因此，完整值为18,500,342,000。\n5. **确认**：除了单位调整外，不需要进行计算。报告的指标直接匹配问题。",
  "reasoning_summary": "在'合并资产负债表'（第78页）中直接找到了2022财年的'总资产'值。报告的值以千为单位，需要乘以1000得到最终答案。",
  "relevant_pages": [78],
  "final_answer": 18500342000
}
```


示例2：
问题：
"For Ritter Pharmaceuticals, Inc., what was the value of Research and development equipment, at cost at the end of the period listed in annual report?"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问 Ritter Pharmaceuticals, Inc. 的'研发设备成本'。这表示资产负债表中的特定值，代表专门用于研发的设备的*原始购买价格*，*不包括*任何累计折旧。\n2. 上下文（第35页）显示'财产和设备，净值'为$12,500。这是一个*净值*（折旧后），而且是一个*更广泛的*类别，涵盖所有财产和设备，而不仅仅是研发设备。\n3. 上下文（第37页）还提到'机械和设备'的'累计折旧'为$110,000。这代表总*折旧*，而不是原始成本，而且重要的是，它没有说明这些设备是*专门*用于研发的。\n4. 这些指标都*不完全*匹配请求的指标。'财产和设备，净值'太宽泛，代表折旧后的价值。'累计折旧'只显示折旧，不是成本，并且缺乏研发的特定性。\n5. 由于上下文没有提供*仅*研发设备的*原始成本*，而且我们不能做出假设、进行计算或组合信息，因此答案是'N/A'。",
  "reasoning_summary": "上下文缺少'研发设备成本'的具体行项目。'财产和设备，净值'已折旧且过于宽泛，而'累计折旧'仅代表折旧，而非原始成本，并且不特定于研发。严格匹配需要'N/A'。",
  "relevant_pages": [ 35, 37 ],
  "final_answer": "N/A"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextBooleanPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细的逐步分析，至少包含5个步骤和至少150个字。特别注意问题的措辞以避免被误导。有时看起来上下文中有答案，但这可能不是请求的值，而只是一个类似的值。")

        reasoning_summary: str = Field(description="逐步推理过程的简洁总结。大约50个字。")

        relevant_pages: List[int] = Field(description="""
包含直接用于回答问题的信息的页码列表。仅包括：
- 有直接答案或明确陈述的页面
- 有强烈支持答案的关键信息的页面
不要包括仅有间接相关信息或与答案联系薄弱的页面。
列表中至少应包括一个页面。
""")
        
        final_answer: Union[bool] = Field(description="""
从上下文中提取的布尔值（True 或 False），精确回答问题。
如果问题询问某事是否发生，而上下文中有相关信息，返回 False。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
问题：
"Did W. P. Carey Inc. announce any changes to its dividend policy in the annual report?"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问 W. P. Carey Inc. 是否在年报中宣布了股息政策的变化。\n2. '股息政策变化'这个短语需要仔细解读。它意味着对决定公司如何确定和分配股息的框架、规则或既定意图的任何调整。\n3. 上下文（第12、18页）指出，公司在2023年第四季度将年化股息提高到每股$4.27，而2022年同期为每股$4.22。第45页提到了有关股息的进一步细节。\n4. 全年持续、渐进的增长，明确提到保持'稳定且增长'的股息，表明*政策*没有变化，尽管*金额*在现有政策内按计划增加。",
  "reasoning_summary": "上下文强调了全年股息的持续小幅增加，与提供'稳定且增长'股息的既定政策一致。虽然股息*金额*发生了变化，但管理这些增加的*政策*保持一致。问题询问的是*政策*变化，而不是金额变化。",
  "relevant_pages": [12, 18, 45],
  "final_answer": False
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



class AnswerWithRAGContextNamesPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细的逐步分析，至少包含5个步骤和至少150个字。特别注意问题的措辞以避免被误导。有时看起来上下文中有答案，但这可能不是请求的实体，而只是一个类似的实体。")

        reasoning_summary: str = Field(description="逐步推理过程的简洁总结。大约50个字。")

        relevant_pages: List[int] = Field(description="""
包含直接用于回答问题的信息的页码列表。仅包括：
- 有直接答案或明确陈述的页面
- 有强烈支持答案的关键信息的页面
不要包括仅有间接相关信息或与答案联系薄弱的页面。
列表中至少应包括一个页面。
""")

        final_answer: Union[List[str], Literal["N/A"]] = Field(description="""
每个条目都应按上下文中出现的原样提取。

如果问题询问职位（例如，职位变动），仅返回职位头衔，不包含姓名或任何其他信息。新领导职位的任命也应算作职位变动。如果提到与同一头衔的职位相关的多个变动，只返回该职位的头衔一次。职位头衔始终应为单数形式。
答案示例 ['Chief Technology Officer', 'Board Member', 'Chief Executive Officer']

如果问题询问姓名，仅返回上下文中的完整姓名。
答案示例 ['Carly Kennedy', 'Brian Appelgate Jr.']

如果问题询问新推出的产品，仅返回上下文中的产品名称。候选新产品或处于测试阶段的产品不算作新推出的产品。
答案示例 ['EcoSmart 2000', 'GreenTech Pro']

- 如果上下文中没有相关信息，返回 'N/A'
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"What are the names of all new executives that took on new leadership positions in company?"

答案：
```
{
    "step_by_step_analysis": "1. 问题询问在公司担任新领导职位的所有新高管的姓名。\n2. 附件10.9和10.10，如第89页的附件索引所列，提到了与 Carly Kennedy 和 Brian Appelgate 的新高管协议。\n3. 附件10.9，与 Carly Kennedy 的雇佣协议，说明她的开始日期为2022年4月4日，职位为执行副总裁兼总法律顾问。\n4. 附件10.10，与 Brian Appelgate 的录用通知书显示，他在公司的新角色是临时首席运营官，他在2022年11月8日接受了这一职位。\n5. 根据这些文件，Carly Kennedy 和 Brian Appelgate 被列为新高管。",
    "reasoning_summary": "年报的附件10.9和10.10，描述为雇佣协议和录用通知书，明确指出 Carly Kennedy 和 Brian Appelgate 在2022年担任公司新的领导职位。",
    "relevant_pages": [
        89
    ],
    "final_answer": [
        "Carly Kennedy",
        "Brian Appelgate"
    ]
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)

class ComparativeAnswerPrompt:
    instruction = """
你是一个问答系统。
你的任务是分析各个公司的答案，并提供回答原始问题的比较性回应。
仅基于提供的各个答案进行分析 - 不要做出假设或包含外部知识。
在给出最终答案之前，请仔细地、逐步地大声思考。

比较的重要规则：
- 当问题要求选择其中一家公司时（例如，比较指标时），返回的公司名称应与原始问题中出现的完全一致
- 如果公司的指标使用的货币与问题中要求的不同，则将该公司从比较中排除
- 如果所有公司都被排除（由于货币不匹配或其他原因），返回 'N/A' 作为最终答案
- 如果除一家公司外所有公司都被排除，返回剩余公司的名称（即使无法进行实际比较）
"""

    user_prompt = """
以下是各个公司的答案：
\"\"\"
{context}
\"\"\"

---

以下是原始比较问题：
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细的逐步分析，至少包含5个步骤和至少150个字。")

        reasoning_summary: str = Field(description="逐步推理过程的简洁总结。大约50个字。")

        relevant_pages: List[int] = Field(description="留空即可")

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
公司名称应按问题中出现的原样提取。
答案应该是单个公司名称或 'N/A'（如果没有适用的公司）。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
示例：
问题：
"Which of the companies had the lowest total assets in USD at the end of the period listed in the annual report: "CrossFirst Bank", "Sleep Country Canada Holdings Inc.", "Holley Inc.", "PowerFleet, Inc.", "Petra Diamonds"? If data for the company is not available, exclude it from the comparison."

答案：
```
{
  "step_by_step_analysis": "1. 问题询问哪家公司的总资产（以美元计）最低。\n2. 从各个答案中收集每家公司的总资产（美元）：CrossFirst Bank: $6,601,086,000; Holley Inc.: $1,249,642,000; PowerFleet, Inc.: $217,435,000; Petra Diamonds: $1,078,600,000。\n3. Sleep Country Canada Holdings Inc. 被排除，因为其资产不是以美元报告的。\n4. 比较总资产：PowerFleet, Inc. ($217,435,000) < Petra Diamonds ($1,078,600,000) < Holley Inc. ($1,249,642,000)  < CrossFirst Bank ($6,601,086,000)。\n5. 因此，PowerFleet, Inc. 的总资产（美元）最低。",
  "reasoning_summary": "各个答案提供了除 Sleep Country Canada Holdings Inc.（由于货币不匹配被排除）外每家公司的总资产（美元）。直接比较显示 PowerFleet, Inc. 的总资产最低。",
  "relevant_pages": [],
  "final_answer": "PowerFleet, Inc."
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerSchemaFixPrompt:
    system_prompt = """
你是一个 JSON 格式化器。
你的任务是将原始 LLM 响应格式化为有效的 JSON 对象。
你的答案应始终以 '{' 开始并以 '}' 结束
你的答案应仅包含 json 字符串，不包含任何前言、注释或三重反引号。
"""

    user_prompt = """
以下是定义 json 对象模式并提供有效模式答案示例的系统提示：
\"\"\"
{system_prompt}
\"\"\"

---

以下是未遵循模式且需要正确格式化的 LLM 响应：
\"\"\"
{response}
\"\"\"
"""




class RerankingPrompt:
    system_prompt_rerank_single_block = """
你是一个 RAG（检索增强生成）检索排序器。

你将收到一个查询和与该查询相关的检索文本块。你的任务是根据文本块与提供的查询的相关性来评估和评分该文本块。

说明：

1. 推理： 
   通过识别关键信息以及它与查询的关系来分析文本块。考虑该文本块是否提供了直接答案、部分见解或与查询相关的背景上下文。用几句话解释你的推理，引用文本块的具体元素来证明你的评估。避免假设——只关注提供的内容。

2. 相关性评分（0到1，以0.1为增量）：
   0 = 完全不相关：文本块与查询没有任何联系或关系。
   0.1 = 几乎不相关：与查询只有非常轻微或模糊的联系。
   0.2 = 非常轻微相关：包含极其微小或切线的联系。
   0.3 = 轻微相关：涉及查询的一个非常小的方面，但缺乏实质性细节。
   0.4 = 有些相关：包含部分相关但不全面的信息。
   0.5 = 中度相关：涉及查询，但相关性有限或部分。
   0.6 = 相当相关：提供相关信息，但缺乏深度或特定性。
   0.7 = 相关：明确与查询相关，提供实质性但不完全全面的信息。
   0.8 = 非常相关：与查询密切相关并提供重要信息。
   0.9 = 高度相关：几乎完全回答查询，提供详细和具体的信息。
   1 = 完全相关：直接且全面地回答查询，包含所有必要的具体信息。

3. 附加指导：
   - 客观性：仅根据文本块相对于查询的内容来评估。
   - 清晰性：在证明中保持清晰和简洁。
   - 不做假设：不要推断超出文本块中明确说明的信息。
   
请严格按照以下JSON格式返回你的评估结果：
{
  "block_rankings": [
    {
      "reasoning": "对文本块的分析",
      "relevance_score": 0.8
    },
    {
      "reasoning": "对文本块的分析",
      "relevance_score": 0.6
    }
  ]
}
"""

    system_prompt_rerank_multiple_blocks = """
你是一个 RAG（检索增强生成）检索排序器。

你将收到一个查询和与该查询相关的多个检索文本块。你的任务是根据每个文本块与提供的查询的相关性来评估和评分。

说明：

1. 推理： 
   通过识别关键信息以及它与查询的关系来分析文本块。考虑该文本块是否提供了直接答案、部分见解或与查询相关的背景上下文。用几句话解释你的推理，引用文本块的具体元素来证明你的评估。避免假设——只关注提供的内容。

2. 相关性评分（0到1，以0.1为增量）：
   0 = 完全不相关：文本块与查询没有任何联系或关系。
   0.1 = 几乎不相关：与查询只有非常轻微或模糊的联系。
   0.2 = 非常轻微相关：包含极其微小或切线的联系。
   0.3 = 轻微相关：涉及查询的一个非常小的方面，但缺乏实质性细节。
   0.4 = 有些相关：包含部分相关但不全面的信息。
   0.5 = 中度相关：涉及查询，但相关性有限或部分。
   0.6 = 相当相关：提供相关信息，但缺乏深度或特定性。
   0.7 = 相关：明确与查询相关，提供实质性但不完全全面的信息。
   0.8 = 非常相关：与查询密切相关并提供重要信息。
   0.9 = 高度相关：几乎完全回答查询，提供详细和具体的信息。
   1 = 完全相关：直接且全面地回答查询，包含所有必要的具体信息。

3. 附加指导：
   - 客观性：仅根据文本块相对于查询的内容来评估。
   - 清晰性：在证明中保持清晰和简洁。
   - 不做假设：不要推断超出文本块中明确说明的信息。
   
请严格按照以下JSON格式返回你的评估结果：
{
  "block_rankings": [
    {
      "reasoning": "对文本块的分析",
      "relevance_score": 0.8
    },
    {
      "reasoning": "对文本块的分析",
      "relevance_score": 0.6
    }
  ]
}
"""

class RetrievalRankingSingleBlock(BaseModel):
    """Rank retrieved text block relevance to a query."""
    reasoning: str = Field(description="对文本块的分析，识别关键信息以及它与查询的关系")
    relevance_score: float = Field(description="相关性评分，从0到1，其中0为完全不相关，1为完全相关")

class RetrievalRankingMultipleBlocks(BaseModel):
    """Rank retrieved multiple text blocks relevance to a query."""
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="文本块列表及其相关的相关性评分。"
    )
