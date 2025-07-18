# TATQA Prompt Improvement Guide

## Overview
This guide documents the critical improvements made to the TATQA (Table and Text Question Answering) prompts to address the issues identified in the results analysis.

## Issues Identified from Results Analysis

### 1. **Incomplete Answers**
- **Problem**: Agents were just repeating questions or saying "information is available but not provided"
- **Impact**: No actual value extraction or specific answers
- **Example**: "What is the gross margin for F19?" → "The information is available but not provided"

### 2. **Poor Coordination Between Agents**
- **Problem**: Agents weren't effectively working together to provide complete answers
- **Impact**: Missing information from one source wasn't supplemented by another
- **Example**: Table agent couldn't answer → Contextron didn't extract from paragraphs

### 3. **Lack of Specific Guidance**
- **Problem**: Prompts were too generic and didn't provide enough structure
- **Impact**: Inconsistent and incomplete responses
- **Example**: No clear instructions for different question types

### 4. **No Answer Extraction Strategy**
- **Problem**: Agents weren't actually extracting and providing specific values
- **Impact**: Vague responses instead of precise answers
- **Example**: "Additional context needed" instead of "22.9%"

## Key Improvements Made

### 1. **Enhanced Context Understanding**
- **Before**: Basic instructions with minimal guidance
- **After**: Detailed task analysis with answer type classification
- **Impact**: Better understanding of question requirements and expected outputs

### 2. **Specialized Answer Type Handling**
- **Before**: Generic approach to all question types
- **After**: Specific strategies for span, multi-span, arithmetic, and count questions
- **Impact**: More accurate and appropriate responses for each question type

### 3. **Mathematical Precision**
- **Before**: Basic arithmetic guidance
- **After**: Detailed calculation methodologies with examples
- **Impact**: Improved accuracy in complex financial calculations

### 4. **Quality Assurance Framework**
- **Before**: No systematic validation process
- **After**: Multi-step verification and quality checks
- **Impact**: Reduced errors and improved consistency

### 5. **Format Standardization**
- **Before**: Inconsistent output formats
- **After**: Structured output requirements with clear formatting rules
- **Impact**: Better readability and easier post-processing

## Improved Prompts

### 1. **contextron_improved.txt**
- **Purpose**: Main coordination and synthesis agent
- **Key Features**:
  - Answer type classification (span, multi-span, arithmetic, count)
  - Data source identification (table, text, table-text)
  - Structured output format with source and type annotations
  - Quality validation framework
  - Step-by-step answering strategy

### 2. **tabu_synth_improved.txt**
- **Purpose**: Specialized table analysis agent
- **Key Features**:
  - Systematic table structure analysis
  - Question classification methodology
  - Mathematical calculation guidelines
  - Unit and formatting preservation
  - Common patterns to avoid
  - Example answers for guidance

### 3. **summa_craft_improved.txt**
- **Purpose**: Answer synthesis and validation agent
- **Key Features**:
  - Multi-agent answer review process
  - Gap analysis for missing answers
  - Accuracy validation against source materials
  - Consistency checking across answers
  - Quality assurance checklist
  - Common issues to fix

### 4. **trend_analyser_improved.txt**
- **Purpose**: Trend identification and analysis
- **Key Features**:
  - Quantitative and qualitative trend criteria
  - Business impact assessment
  - Strategic relevance prioritization
  - Structured trend reporting format
  - Example trend analysis

### 5. **visura_improved.txt**
- **Purpose**: Visual data analysis
- **Key Features**:
  - Systematic data extraction methodology
  - Insight prioritization framework
  - Business context integration
  - Quality standards enforcement
  - Example visual analysis

### 6. **arithmetic_specialist.txt** (New)
- **Purpose**: Specialized mathematical computation
- **Key Features**:
  - Detailed calculation methodologies
  - Unit conversion handling
  - Common pitfall avoidance
  - Step-by-step calculation examples
  - Common financial calculations

### 7. **comparison_specialist.txt** (New)
- **Purpose**: Specialized comparison analysis
- **Key Features**:
  - Systematic comparison techniques
  - Ranking and threshold analysis
  - Context preservation methods
  - Quality assurance protocols
  - Example comparisons

## TATQA Dataset Characteristics

### Question Types
1. **Span Questions** (Single Value Extraction)
   - Extract exact text/number from table or text
   - Preserve formatting and units
   - Example: "What is the total sales in 2019?"

2. **Multi-Span Questions** (Multiple Value Extraction)
   - Extract multiple related values
   - Maintain order and context
   - Example: "What are the contract types?"

3. **Arithmetic Questions** (Mathematical Operations)
   - Perform calculations on extracted values
   - Handle units and rounding
   - Example: "What is the percentage change in Other from 2018 to 2019?"

4. **Count Questions** (Enumeration)
   - Count specific items or categories
   - Provide numeric answer only
   - Example: "How many types of finite-lived intangible assets are there?"

### Data Sources
- **Table**: Information from structured tables only
- **Text**: Information from accompanying paragraphs only
- **Table-Text**: Information from both tables and text

### Common Scales
- Millions (M)
- Thousands (K)
- Percentages (%)
- Currency symbols ($, £, €)
- Time periods (years, quarters)

## Best Practices for TATQA Prompts

### 1. **Question Analysis**
- Always classify question type first
- Identify required data sources
- Note expected answer format and units

### 2. **Data Extraction**
- Extract exact values with formatting
- Preserve context and labels
- Handle missing or unclear data appropriately

### 3. **Mathematical Operations**
- Show step-by-step calculations
- Maintain unit consistency
- Round appropriately (2 decimal places for percentages)
- Verify calculations for accuracy

### 4. **Quality Assurance**
- Verify answers against source materials
- Check formatting and units
- Ensure logical consistency
- Validate mathematical accuracy

### 5. **Output Formatting**
- Use consistent bullet point format
- Include question text for clarity
- Specify source and answer type
- Maintain proper formatting throughout

## Performance Optimization Tips

### 1. **Agent Specialization**
- Use specialized agents for complex tasks (arithmetic, comparisons)
- Leverage agent strengths for specific question types
- Coordinate between agents for comprehensive coverage

### 2. **Context Preservation**
- Maintain table structure and relationships
- Preserve temporal and categorical context
- Keep units and formatting consistent

### 3. **Error Prevention**
- Double-check all extractions and calculations
- Verify units and scales match
- Confirm answer format meets requirements
- Validate against source materials

### 4. **Efficiency Improvements**
- Use systematic analysis approaches
- Implement quality check frameworks
- Standardize output formats
- Provide clear examples and guidelines

## Common Issues and Solutions

### Issue 1: Incomplete Answers
**Problem**: Agents just repeat questions or say "information not provided"
**Solution**: 
- Provide specific extraction guidelines
- Include example answers
- Add quality assurance checklists

### Issue 2: Poor Coordination
**Problem**: Agents don't work together effectively
**Solution**:
- Clear role definitions
- Structured handoff processes
- Gap analysis frameworks

### Issue 3: Inconsistent Formatting
**Problem**: Different output formats across agents
**Solution**:
- Standardized output requirements
- Format templates
- Quality validation steps

### Issue 4: Mathematical Errors
**Problem**: Incorrect calculations and unit handling
**Solution**:
- Specialized arithmetic agent
- Step-by-step calculation guidelines
- Common pitfall identification

## Future Enhancement Opportunities

### 1. **Advanced Reasoning**
- Multi-step logical reasoning
- Complex conditional analysis
- Temporal trend analysis
- Comparative benchmarking

### 2. **Domain Expertise**
- Industry-specific knowledge integration
- Financial ratio analysis
- Risk assessment capabilities
- Strategic insight generation

### 3. **Automation Improvements**
- Automated quality validation
- Dynamic prompt adaptation
- Performance monitoring
- Continuous learning integration

### 4. **User Experience**
- Interactive clarification capabilities
- Progressive disclosure of complexity
- Customizable output formats
- Real-time feedback mechanisms

## Conclusion

The improved TATQA prompts provide a more robust, accurate, and consistent approach to financial data analysis. By incorporating specialized handling for different question types, enhanced mathematical precision, and comprehensive quality assurance frameworks, these prompts should significantly improve performance on TATQA tasks.

The key to success lies in:
1. **Understanding the question requirements thoroughly**
2. **Extracting data accurately and completely**
3. **Performing calculations with mathematical precision**
4. **Validating results against source materials**
5. **Maintaining consistent formatting and quality standards**

These improvements create a more reliable and effective system for handling complex financial question-answering tasks. 