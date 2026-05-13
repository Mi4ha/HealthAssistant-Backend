# RAG 知识库说明

本目录存放 HealthAssistant 后端 RAG 使用的结构化 Markdown 知识库。

## 目录分层

- `general/`：默认通用知识库，面向普通个人健康管理用户，默认参与检索。
- `risk_management/`：风险管理知识库，目前主要用于体重管理。仅在 BMI/问题语义触发时参与检索。
- `condition_specific/`：疾病专项知识库。仅当用户明确提到对应疾病、指标或后续健康档案存在对应字段时启用，不参与普通默认问答。

## 来源原则

知识条目基于本地 `source_docs/` 中下载或保存的权威来源整理，包括中国营养学会、国家卫生健康委、中华流行病学杂志等来源。每个 Markdown 文件的 YAML 元数据中包含 `source`、`source_url`、`source_local_files`、`knowledge_scope` 和 `activation_rule` 等字段，便于后端检索和答辩说明。

## 使用边界

系统定位是健康管理和健康教育助手，不做诊断，不替代医生，不给药物剂量或治疗方案。疾病专项指南仅作为条件触发的健康教育依据。
