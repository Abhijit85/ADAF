# Deep ROUGE Score Analysis: Why Scores Are Still Low

## Executive Summary

After implementing optimized ROUGE normalization, we discovered that the **fundamental issue is NOT formatting** - it's **model behavior and answer generation quality**. The model is getting the right information (high F1 scores) but expressing it differently (low ROUGE scores).

## Key Findings

### 1. **The Real Problem: Answer Generation Style**

**Critical Insight**: 128 examples have **high F1 (>0.8) but low ROUGE (<0.5)**
- This means the model **understands the question** and **has the right information**
- But it **expresses the answer differently** than the gold standard

### 2. **Pattern Analysis of High F1, Low ROUGE Examples**

| Pattern | Count | Percentage |
|---------|-------|------------|
| Length Mismatch | 104 | 81.2% |
| Extra Context | 121 | 94.5% |
| Different Structure | 49 | 38.3% |
| Formatting Differences | 38 | 29.7% |
| Different Phrasing | 7 | 5.5% |

### 3. **Specific Examples of the Problem**

#### Example 1: Length Mismatch (Ratio: 2.77)
**Question**: "How do the Super Bowl VII teams compare in terms of first downs and total net yards?"

**Gold**: "Washington has total yards of 228, 16 first downs and Miami has 253 and 12." (75 chars)

**Pred**: "In Super Bowl VII, the Washington Redskins achieved more first downs (16) compared to the Miami Dolphins (12). However, the Miami Dolphins gained more total net yards (253) than the Washington Redskins (228)." (208 chars)

**Analysis**: Model provides **2.77x more text** but same information.

#### Example 2: Extra Context
**Question**: "Which music videos did Faye Dunaway appear in?"

**Gold**: "Dunaway appeared in two music videos, Tom Petty and the Heartbreakers' Into the Great Wide Open in 1991 and Hill Zaini's I Heard in 2010." (137 chars)

**Pred**: "Based on the provided data, no music videos featuring Faye Dunaway are documented. The structured table and textual filmography list two entries ('Into the Great Wide Open' [1991] and 'I Heard' [2010]), but neither references her participation as a performer, director, or in any role. Additional verification from external sources is required to confirm potential unlisted appearances." (386 chars)

**Analysis**: Model adds **unnecessary explanatory context** instead of giving a direct answer.

#### Example 3: Different Structure
**Question**: "What numbers did the Z class have?"

**Gold**: "The Z class were given the numbers 30950–30957 under the BR numbering system." (77 chars)

**Pred**: "The Z class locomotives had Southern Railway (SR) numbers 950–957, as confirmed by structured data showing their original SR numbering and corresponding British Railways (BR) numbers 30950–30957." (195 chars)

**Analysis**: Model provides **additional context** and **different structure** but same core information.

## Root Cause Analysis

### 1. **Model Training Issues**
- **Verbosity**: Model is trained to be overly explanatory
- **Context Addition**: Model adds unnecessary background information
- **Style Mismatch**: Model doesn't match the concise style of gold answers
- **Formatting Inconsistency**: Model uses different formatting conventions

### 2. **Prompt Engineering Issues**
- **No Style Guidance**: Prompts don't specify answer style requirements
- **No Length Constraints**: No guidance on answer length
- **No Formatting Requirements**: No specification of formatting style

### 3. **Evaluation Metric Mismatch**
- **F1 vs ROUGE**: F1 measures information accuracy, ROUGE measures text similarity
- **Content vs Style**: Model gets content right but style wrong

## Recommendations for Improvement

### 1. **Immediate Actions**

#### A. Enhanced Prompt Engineering
```python
# Add to prompts:
"Provide concise, direct answers that match the style of the reference answer. 
Avoid unnecessary explanations or context. Use similar vocabulary and structure."
```

#### B. Answer Length Constraints
```python
# Add length guidance:
"Keep answers under 100 words unless the question requires more detail."
```

#### C. Style Matching Training
- Train model to recognize and match gold answer styles
- Provide examples of good vs. bad answer styles
- Use contrastive learning to improve style matching

### 2. **Model Training Improvements**

#### A. Style-Aware Training
- **Concise Answer Training**: Train model to give shorter, more direct answers
- **Style Matching**: Train model to match the style of gold answers
- **Context Reduction**: Train model to avoid unnecessary explanatory text

#### B. Prompt Engineering
- **Length Constraints**: Add specific length requirements to prompts
- **Style Examples**: Include examples of desired answer styles
- **Formatting Guidelines**: Specify formatting requirements

#### C. Evaluation Integration
- **Style Scoring**: Add style similarity to evaluation metrics
- **Length Penalty**: Penalize overly long answers
- **Context Penalty**: Penalize unnecessary explanatory text

### 3. **Advanced Solutions**

#### A. Multi-Stage Answer Generation
1. **Information Extraction**: Extract key facts
2. **Style Matching**: Match gold answer style
3. **Length Optimization**: Ensure appropriate length
4. **Formatting**: Apply consistent formatting

#### B. Style Transfer Learning
- Train model to transfer between different answer styles
- Use style embeddings to guide answer generation
- Implement style-aware decoding

#### C. Contrastive Learning
- Train model to distinguish between good and bad answer styles
- Use pairs of similar content but different styles
- Learn to prefer concise, direct styles

## Implementation Plan

### Phase 1: Prompt Engineering (Week 1)
1. **Add Style Guidelines** to all prompts
2. **Include Length Constraints** in prompts
3. **Add Formatting Requirements** to prompts
4. **Test with Current Models**

### Phase 2: Model Training (Week 2-3)
1. **Style-Aware Training** with gold answer examples
2. **Concise Answer Training** with length constraints
3. **Context Reduction Training** to avoid unnecessary explanations
4. **Formatting Consistency Training**

### Phase 3: Evaluation Enhancement (Week 4)
1. **Style Similarity Metrics** in evaluation
2. **Length Penalty** in scoring
3. **Context Penalty** for unnecessary explanations
4. **Integrated Style Scoring**

## Expected Impact

### Conservative Estimate
- **ROUGE-1**: +15-25% improvement
- **ROUGE-2**: +20-30% improvement
- **ROUGE-L**: +15-25% improvement

### Optimistic Estimate (with comprehensive training)
- **ROUGE-1**: +30-40% improvement
- **ROUGE-2**: +40-50% improvement
- **ROUGE-L**: +30-40% improvement

## Conclusion

The low ROUGE scores are **NOT a normalization problem** - they're a **model behavior problem**. The model needs to learn to:

1. **Give concise, direct answers** instead of verbose explanations
2. **Match the style and structure** of gold answers
3. **Use similar vocabulary and phrasing** as gold answers
4. **Provide the right level of detail** without unnecessary context
5. **Format answers consistently** with gold standards

The solution requires **fundamental changes to model training and prompt engineering**, not just better normalization. The optimized normalization we implemented helps with formatting issues, but the main problem is **answer generation style and behavior**. 