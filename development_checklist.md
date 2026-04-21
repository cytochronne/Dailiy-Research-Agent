Daily arXiv Research Briefing Agent 项目需求
1. 项目目标

构建一个 Daily arXiv Research Briefing Agent，能够每日获取 arXiv 新论文，完成个性化推荐、结构化简报生成，以及论文级深入讲解。整体系统需采用 Agent + Skills 架构：每个 Skill 功能单一、输入输出清晰、可独立测试，最终由 Agent 编排成完整流程。

2. 基本需求（guidance 要求）
2.1 每日 arXiv 监测与获取

系统需要能够按日获取 arXiv 新论文。

需要完成：

支持按日期/时间范围抓取新论文
支持按 topic / keyword / arXiv category 过滤
保存基础元数据：title、authors、abstract、category、date、url
2.2 论文相关性排序

系统需要对获取到的新论文进行排序。

需要完成：

根据用户输入的研究方向或关键词进行 relevance ranking
输出 Top-K 推荐结果
每篇论文应带有相关性分数或排序依据
2.3 关键信息抽取

系统需要从论文中抽取结构化信息。

需要完成：

提取 key contributions
提取 methods
生成简明摘要
输出统一结构字段，供后续 briefing 使用
2.4 Daily Briefing 生成

系统需要生成结构化日报。

需要完成：

输出 summary table
输出推荐论文列表
输出每篇论文的简要介绍
高亮最相关/最值得关注的论文
2.5 Follow-up queries

系统需要支持后续查询。

需要完成：

支持按 topic 继续筛选
支持按时间范围继续筛选
支持类似：
“show me papers related to graph neural networks from this week”
“只看本周的某类论文”
3. 扩展需求 1：Personalized recommendation without requiring explicit keywords and interactive preference refinement
3.1 无需显式关键词的个性化推荐

用户不一定输入 topic/keywords，而是上传几篇感兴趣的论文，系统据此进行推荐。

需要完成：

支持用户上传若干 seed papers
输入形式可限定为：arXiv ID / URL / title / PDF
解析 seed papers 的基础内容
基于 seed papers 构建用户兴趣表示
对新论文进行个性化排序，而不是只依赖显式关键词
3.2 点赞/点踩反馈闭环

用户可以对推荐结果进行 like / dislike，系统需要据此调整后续推荐。

需要完成：

在推荐结果中支持用户反馈记录
根据 like / dislike 更新偏好表示
支持多轮推荐 refinement
新一轮推荐结果应体现反馈后的偏好变化
3.3 这一部分的系统输出

需要输出：

seed-paper-based recommendation list
updated recommendation list after feedback
每轮推荐的排序结果和变化依据
4. 扩展需求 2：Paper-level deep explanation

用户在推荐结果中选中某篇论文后，系统需要支持更深入的讲解，而不是只给短摘要。

4.1 论文级深入讲解入口

需要完成：

用户可从推荐列表中选择某篇论文
用户可选择讲解模式
系统根据模式调用对应解释逻辑
4.2 支持的讲解模式

至少支持以下三类：

1）方法框架详细讲解

需要完成：

解释文章解决什么问题
解释方法整体框架
解释核心模块、输入输出、流程
说明创新点在哪里
2）实验设置与实验结果

需要完成：

解释用了哪些数据集
解释 baseline / metrics
解释实验设置
总结主要实验结果与结论
3）文章局限性

需要完成：

总结文章已有局限
分析隐含假设
指出可能缺失的验证或风险点
4.3 深入讲解的数据基础

开发需要明确：

是只基于 abstract
还是读取全文/PDF 后再解释

若要支持实验和局限性讲解，推荐至少支持正文级内容解析。

5. 系统架构需求
5.1 必须采用 Agent + Skills 设计

需要完成：

每个 Skill 功能独立
每个 Skill 输入输出清晰
每个 Skill 可单独测试
Agent 负责统一编排流程
5.2 建议的 Skill 划分

开发至少应覆盖以下功能模块：

arXiv Retrieval Skill
获取新论文与元数据
Seed Paper Parsing / Preference Modeling Skill
解析用户上传论文
构建兴趣表示
Personalized Ranking Skill
基于关键词或 seed papers 进行排序
Feedback Update Skill
处理 like / dislike
更新推荐偏好
Summarization / Briefing Skill
抽取 contributions、methods、summary
生成日报
Paper Deep Explanation Skill
对单篇论文进行细讲
按方法/实验/局限性三种模式输出
Agent Orchestrator
控制整体调用顺序
整合最终结果
6. 端到端工作流需求
6.1 推荐工作流

需要实现：

用户输入 topic/keywords，或上传 seed papers
系统获取当日/近期 arXiv 新论文
系统进行相关性/个性化排序
系统抽取关键信息
系统生成 daily briefing
用户对推荐结果点赞/点踩
系统更新偏好并输出 refined recommendations
6.2 深入讲解工作流

需要实现：

用户从推荐列表中选中某篇论文
用户选择讲解模式
系统生成对应的详细解释
7. 输入与输出要求
输入

系统需支持以下输入：

research topic / keywords
date range
seed papers
user feedback: like / dislike
selected paper
selected explanation mode
输出

系统需支持以下输出：

ranked paper list
structured daily briefing
recommendation updates after feedback
detailed explanation for selected paper
8. 评测与交付要求
8.1 评测

开发需要预留评测能力，至少支持：

推荐质量评测
反馈前后推荐变化评测
深入讲解质量评测
8.2 可视化/展示

项目最终需要能展示：

Agent workflow
推荐结果
反馈闭环效果
单篇论文深入讲解示例
8.3 报告与代码要求

最终交付需满足课程要求：

Group report（Agent）
Individual report（Skill）
Agent 与 Skills 代码提交/发布
presentation 展示
最终开发清单

下游开发至少需要完成以下内容：

新论文抓取模块
关键词检索与排序模块
seed papers 上传与解析模块
个性化兴趣建模模块
点赞/点踩反馈更新模块
论文摘要与结构化信息抽取模块
daily briefing 生成模块
follow-up query 模块
paper-level deep explanation 模块
Agent orchestration 与整体接口整合