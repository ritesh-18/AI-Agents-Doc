# Agentic Design Patterns — Easy Language Guide

> **Book:** *Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems* by Antonio Gulli  
> **Purpose:** A plain-English summary of all 21 patterns, with JavaScript code examples and book diagrams.

---

## Table of Contents

1. [What is an AI Agent?](#what-is-an-ai-agent)
2. [Why Design Patterns Matter](#why-design-patterns-matter)
3. [Part 1 — Core Patterns](#part-1--core-patterns)
   - [Chapter 1: Prompt Chaining](#chapter-1-prompt-chaining)
   - [Chapter 2: Routing](#chapter-2-routing)
   - [Chapter 3: Parallelization](#chapter-3-parallelization)
   - [Chapter 4: Reflection](#chapter-4-reflection)
   - [Chapter 5: Tool Use (Function Calling)](#chapter-5-tool-use-function-calling)
   - [Chapter 6: Planning](#chapter-6-planning)
   - [Chapter 7: Multi-Agent Collaboration](#chapter-7-multi-agent-collaboration)
4. [Part 2 — Memory & Adaptation](#part-2--memory--adaptation)
   - [Chapter 8: Memory Management](#chapter-8-memory-management)
   - [Chapter 9: Learning and Adaptation](#chapter-9-learning-and-adaptation)
   - [Chapter 10: Model Context Protocol (MCP)](#chapter-10-model-context-protocol-mcp)
   - [Chapter 11: Goal Setting and Monitoring](#chapter-11-goal-setting-and-monitoring)
5. [Part 3 — Robustness & Safety](#part-3--robustness--safety)
   - [Chapter 12: Exception Handling and Recovery](#chapter-12-exception-handling-and-recovery)
   - [Chapter 13: Human-in-the-Loop](#chapter-13-human-in-the-loop)
   - [Chapter 14: Knowledge Retrieval (RAG)](#chapter-14-knowledge-retrieval-rag)
6. [Part 4 — Advanced Patterns](#part-4--advanced-patterns)
   - [Chapter 15: Inter-Agent Communication (A2A)](#chapter-15-inter-agent-communication-a2a)
   - [Chapter 16: Resource-Aware Optimization](#chapter-16-resource-aware-optimization)
   - [Chapter 17: Reasoning Techniques](#chapter-17-reasoning-techniques)
   - [Chapter 18: Guardrails / Safety Patterns](#chapter-18-guardrailssafety-patterns)
   - [Chapter 19: Evaluation and Monitoring](#chapter-19-evaluation-and-monitoring)
   - [Chapter 20: Prioritization](#chapter-20-prioritization)
   - [Chapter 21: Exploration and Discovery](#chapter-21-exploration-and-discovery)
7. [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## What is an AI Agent?

An **AI Agent** is like a smart assistant that doesn't just answer one question — it can take a goal, make a plan, use tools, and keep going until the job is done.

Think of it like this: if you ask a regular AI chatbot "organize my schedule," it gives you advice. An AI agent actually *does* it — it reads your emails, checks your calendar, sends invites, and reschedules conflicts.

### The 5-Step Agent Loop

![Agent Loop](agentic_images/agent_loop.png)

Every agent follows a simple loop:

1. **Get the Mission** — You give it a goal (e.g., "organize my schedule")
2. **Scan the Scene** — It gathers info: reads emails, checks calendars
3. **Think It Through** — It plans the best approach
4. **Take Action** — Sends invites, updates calendar
5. **Learn and Improve** — If a meeting gets rescheduled, it learns and adapts

### Levels of Agent Complexity

![Agent Levels](agentic_images/agent_levels.png)

| Level | What It Can Do | Example |
|-------|---------------|---------|
| **Level 0** — Core Reasoner | Answers from training data only | Explaining history |
| **Level 1** — Connected Solver | Uses tools to get external data | Looking up today's stock price |
| **Level 2** — Strategic Solver | Plans multi-step tasks, engineers context | Travel assistant booking a trip |
| **Level 3** — Multi-Agent System | Team of specialists collaborating | Product launch with research, design, and marketing agents |

### The Future of Agents

![Future Hypotheses](agentic_images/agent_future.png)

Five big predictions for where agents are heading:

1. **Generalist Agents** — One agent that can manage any complex goal for weeks
2. **Deep Personalization** — Agents that *know* you and proactively help
3. **Physical World Agents** — Robots that can fix your leaky tap
4. **Agent Economy** — Agents running entire businesses autonomously
5. **Self-Designing Systems** — Agent teams that rewrite their own code to get better

---

## Why Design Patterns Matter

Building an agent is complex. Without proven patterns, developers waste time reinventing solutions to common problems. These 21 patterns are like **proven blueprints** for the most common challenges in agent design.

> Just as a building architect uses known patterns (arches, load-bearing walls) so they don't have to figure out physics from scratch, agent developers use these patterns to build reliable, intelligent systems faster.

---

## Part 1 — Core Patterns

---

### Chapter 1: Prompt Chaining

**The simple idea:** Instead of asking an AI to do everything in one giant request (which often fails), break the task into a *chain* of smaller steps where each output feeds the next.

**Real-world analogy:** Imagine writing a report. You don't write the whole thing at once — you first research, then outline, then draft, then edit. Each step builds on the last.

![Prompt Chaining](agentic_images/prompt_chaining.png)

**Why single prompts fail for complex tasks:**
- The AI loses track of early instructions (context drift)
- Errors in one part spread (error propagation)
- The AI hallucinates when overwhelmed

**How chaining fixes this:**

Instead of one massive prompt like "analyze this report, find trends, and write an email," you break it into:

1. `Prompt 1:` "Summarize the report" → get summary
2. `Prompt 2:` "From this summary, identify the top 3 trends" → get trends as JSON
3. `Prompt 3:` "Write an email about these trends" → final email

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function promptChain(documentText) {
  // Step 1: Summarize the document
  const summaryResponse = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "You are a market analyst. Summarize the following document." },
      { role: "user", content: documentText }
    ]
  });
  const summary = summaryResponse.choices[0].message.content;
  console.log("Step 1 - Summary:", summary);

  // Step 2: Extract trends from the summary (structured JSON output)
  const trendsResponse = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: "Extract the top 3 market trends as a JSON array: [{\"trend\": \"...\", \"evidence\": \"...\"}]"
      },
      { role: "user", content: `Summary: ${summary}` }
    ]
  });
  const trends = JSON.parse(trendsResponse.choices[0].message.content);
  console.log("Step 2 - Trends:", trends);

  // Step 3: Draft an email from the extracted trends
  const emailResponse = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: "Write a professional email to the marketing team about these trends."
      },
      { role: "user", content: JSON.stringify(trends) }
    ]
  });
  const email = emailResponse.choices[0].message.content;
  console.log("Step 3 - Final Email:", email);
  return email;
}

promptChain("Q3 sales grew 12% driven by Gen Z consumers preferring eco-friendly products...");
```

**Key Takeaways:**
- Break complex tasks into focused steps
- Each step's output becomes the next step's input
- Use structured formats (JSON) between steps to avoid ambiguity
- Use this pattern whenever a task has **multiple distinct stages**

---

### Chapter 2: Routing

**The simple idea:** Not all requests are the same. A smart agent should look at what you're asking and send it to the *right specialist* rather than trying to handle everything the same way.

**Real-world analogy:** When you call a company, the phone system routes you — press 1 for billing, press 2 for tech support. The AI does this automatically based on understanding your request.

![Routing](agentic_images/routing.png)

**How routing works:**

1. User sends a message
2. A **router** (could be the LLM itself) classifies the intent
3. The request is directed to the best handler

**Types of routers:**
- **LLM-based routing** — Ask the model: "Is this a booking, info, or unclear request? Reply with one word."
- **Rule-based routing** — Simple if/else: if message contains "book", go to booking handler
- **Embedding-based routing** — Find the most *semantically similar* route using vector math

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Define handlers for different routes
function bookingHandler(request) {
  console.log("BOOKING HANDLER:", request);
  return `Booking confirmed for: "${request}"`;
}

function infoHandler(request) {
  console.log("INFO HANDLER:", request);
  return `Here is information about: "${request}"`;
}

function unclearHandler(request) {
  return `Sorry, I couldn't understand: "${request}". Could you clarify?`;
}

// The router uses the LLM to classify the intent
async function classifyRequest(userMessage) {
  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `Classify the user request into one of three categories:
          - "booking" (if it's about booking flights, hotels, or appointments)
          - "info" (if it's a general knowledge question)
          - "unclear" (if you can't determine the intent)
          Reply with ONLY one word: booking, info, or unclear.`
      },
      { role: "user", content: userMessage }
    ]
  });
  return response.choices[0].message.content.trim().toLowerCase();
}

// Main routing function
async function routeRequest(userMessage) {
  const intent = await classifyRequest(userMessage);
  console.log(`Detected intent: "${intent}" for message: "${userMessage}"`);

  switch (intent) {
    case "booking":
      return bookingHandler(userMessage);
    case "info":
      return infoHandler(userMessage);
    default:
      return unclearHandler(userMessage);
  }
}

// Test different types of requests
async function main() {
  await routeRequest("Book me a flight to London");        // → booking handler
  await routeRequest("What is the capital of Australia?"); // → info handler
  await routeRequest("Do the thing with the stuff");       // → unclear handler
}

main();
```

**Key Takeaways:**
- Routing makes agents dynamic and context-aware
- LLMs can classify intent very accurately
- Use routing when different types of requests need different handling
- It's the foundation for building multi-purpose assistants

---

### Chapter 3: Parallelization

**The simple idea:** Instead of doing tasks one after another (A → B → C), do them *at the same time* (A + B + C simultaneously), then combine results.

**Real-world analogy:** When cooking, you don't boil the pasta, *then* make the sauce, *then* prep the salad. You do them all at once to save time.

![Parallelization](agentic_images/parallelization.png)

**When to use parallelization:**
- Fetching data from multiple APIs at once
- Running different analysis on the same data
- Generating multiple content variations to compare

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function analyzeTopicInParallel(topic) {
  console.log(`Analyzing "${topic}" with 3 parallel tasks...`);

  // Define 3 independent tasks that can run at the same time
  const summaryTask = client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "Summarize the topic in 2 sentences." },
      { role: "user", content: topic }
    ]
  });

  const questionsTask = client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "Generate 3 interesting questions about this topic." },
      { role: "user", content: topic }
    ]
  });

  const keyTermsTask = client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "List 5 key terms related to this topic, comma-separated." },
      { role: "user", content: topic }
    ]
  });

  // Run all 3 tasks simultaneously using Promise.all
  const [summaryRes, questionsRes, keyTermsRes] = await Promise.all([
    summaryTask,
    questionsTask,
    keyTermsTask
  ]);

  const summary = summaryRes.choices[0].message.content;
  const questions = questionsRes.choices[0].message.content;
  const keyTerms = keyTermsRes.choices[0].message.content;

  // Final synthesis step (sequential, after all parallel tasks finish)
  const synthesisResponse = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: "Synthesize the following into a comprehensive overview."
      },
      {
        role: "user",
        content: `Summary: ${summary}\n\nKey Questions: ${questions}\n\nKey Terms: ${keyTerms}\n\nTopic: ${topic}`
      }
    ]
  });

  console.log("Final Overview:", synthesisResponse.choices[0].message.content);
}

analyzeTopicInParallel("The impact of renewable energy on global economies");
```

**Key Takeaways:**
- Use `Promise.all()` in JavaScript to run tasks simultaneously
- Dramatically reduces total time for multi-step workflows
- Only parallelize tasks that are **independent** (don't need each other's output)
- Final synthesis step remains sequential (waits for all parallel results)

---

### Chapter 4: Reflection

**The simple idea:** After generating an answer, the agent looks back at its own work, critiques it, and improves it — like having an editor review your first draft.

**Real-world analogy:** A lawyer doesn't submit the first draft of a brief. They write it, review it against the requirements, find weaknesses, revise, and repeat until it's solid.

![Reflection](agentic_images/reflection.png)

**Two models of reflection:**

1. **Self-reflection** — The same AI reviews its own output
2. **Producer-Critic** — One AI generates, a *different* AI criticizes (more objective)

The Producer-Critic model is more powerful because the critic has a fresh perspective and specific instructions to find flaws.

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function reflectionLoop(task, maxIterations = 3) {
  let currentCode = "";
  
  for (let i = 0; i < maxIterations; i++) {
    console.log(`\n=== Iteration ${i + 1} ===`);
    
    // PRODUCER: Generate or refine code
    const producerPrompt = i === 0
      ? `Write a JavaScript function for: ${task}`
      : `Improve this JavaScript function based on the critique:\n\n${currentCode}`;
    
    const producerResponse = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are an expert JavaScript developer. Write clean, well-commented code." },
        { role: "user", content: producerPrompt }
      ]
    });
    currentCode = producerResponse.choices[0].message.content;
    console.log("Generated Code:\n", currentCode);
    
    // CRITIC: Review the code
    const criticResponse = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a senior code reviewer. Evaluate if this code:
            1. Is functionally correct
            2. Handles edge cases (null, empty input, invalid types)
            3. Has clear variable names and comments
            4. Is efficient
            
            If everything is perfect, respond with exactly: "CODE_IS_PERFECT"
            Otherwise, list specific issues as bullet points.`
        },
        {
          role: "user",
          content: `Task: ${task}\n\nCode to review:\n${currentCode}`
        }
      ]
    });
    
    const critique = criticResponse.choices[0].message.content;
    console.log("Critique:", critique);
    
    // Check stopping condition
    if (critique.includes("CODE_IS_PERFECT")) {
      console.log("\n✅ Code approved after", i + 1, "iteration(s)!");
      break;
    }
  }
  
  return currentCode;
}

// Test: Write a function to calculate factorial with edge case handling
reflectionLoop("calculate factorial of a number, handling edge cases like 0 and negative numbers");
```

**Key Takeaways:**
- Reflection creates a quality improvement loop
- Producer-Critic separation gives more objective feedback
- Use it when output quality matters more than speed
- Best for: writing code, creating long-form content, planning, debugging

---

### Chapter 5: Tool Use (Function Calling)

**The simple idea:** An AI's knowledge is frozen at training time. Tool use lets an agent "reach out" and call real functions — check live weather, query databases, send emails — giving it superpowers beyond its training data.

**Real-world analogy:** A brilliant doctor with no internet connection still can't tell you today's news. Tool use is like giving that doctor a phone so they can look things up, make calls, and take real actions.

![Tool Use](agentic_images/tool_use.png)

**How it works:**
1. You describe available tools/functions to the AI (name, what it does, what parameters it takes)
2. The AI decides *when* to use a tool based on the user's request
3. The AI generates a structured call: `{ function: "get_weather", args: { location: "London" } }`
4. Your code runs the actual function and returns the result
5. The AI uses that result to form its final answer

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Define your actual tool functions
const tools = {
  get_weather: async ({ location }) => {
    // In real life, call a weather API here
    const mockData = {
      "london": { temp: "15°C", condition: "Cloudy" },
      "paris": { temp: "20°C", condition: "Sunny" },
    };
    return mockData[location.toLowerCase()] || { temp: "Unknown", condition: "Data not available" };
  },
  
  get_stock_price: async ({ ticker }) => {
    // In real life, call a stock API here
    const mockPrices = { "AAPL": 178.50, "GOOGL": 175.20, "MSFT": 425.00 };
    const price = mockPrices[ticker.toUpperCase()];
    return price ? { ticker, price, currency: "USD" } : { error: "Ticker not found" };
  }
};

// Tool definitions for the AI to understand what's available
const toolDefinitions = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get current weather for a given city",
      parameters: {
        type: "object",
        properties: {
          location: { type: "string", description: "The city name, e.g. London" }
        },
        required: ["location"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_stock_price",
      description: "Get the current stock price for a given ticker symbol",
      parameters: {
        type: "object",
        properties: {
          ticker: { type: "string", description: "Stock ticker symbol, e.g. AAPL" }
        },
        required: ["ticker"]
      }
    }
  }
];

async function agentWithTools(userMessage) {
  const messages = [
    { role: "user", content: userMessage }
  ];
  
  // First call: AI decides if it needs a tool
  let response = await client.chat.completions.create({
    model: "gpt-4o",
    messages,
    tools: toolDefinitions,
    tool_choice: "auto"
  });
  
  let assistantMessage = response.choices[0].message;
  
  // If AI called a tool, execute it and continue
  if (assistantMessage.tool_calls) {
    messages.push(assistantMessage); // Add AI's tool request to history
    
    for (const toolCall of assistantMessage.tool_calls) {
      const toolName = toolCall.function.name;
      const toolArgs = JSON.parse(toolCall.function.arguments);
      
      console.log(`Calling tool: ${toolName} with args:`, toolArgs);
      
      const toolResult = await tools[toolName](toolArgs);
      
      // Add tool result to the conversation
      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify(toolResult)
      });
    }
    
    // Final call: AI uses tool results to form the answer
    response = await client.chat.completions.create({
      model: "gpt-4o",
      messages
    });
    assistantMessage = response.choices[0].message;
  }
  
  console.log("Final Answer:", assistantMessage.content);
  return assistantMessage.content;
}

// Test
agentWithTools("What's the weather in London and how much is Apple stock right now?");
```

**Key Takeaways:**
- Tools break the LLM out of its "frozen knowledge" bubble
- The LLM decides *when* to call a tool — you don't hardcode it
- Common tools: search, databases, email, weather APIs, calculators, code runners
- This pattern is foundational — most agents use tool calling

---

### Chapter 6: Planning

**The simple idea:** For complex goals, don't just start doing things randomly. First make a *plan* — break the big goal into ordered steps, then execute them.

**Real-world analogy:** A project manager doesn't just start a project — they write a plan with phases, dependencies, and milestones first.

![Planning](agentic_images/planning.png)

**What makes planning powerful:**
- The agent *dynamically discovers* the how, not just executes the what
- Plans adapt when obstacles arise (a venue is unavailable → find another)
- Turns a single request into a coordinated series of actions

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function planningAgent(goal) {
  console.log(`Goal: ${goal}\n`);
  
  // Step 1: Create a plan
  const planResponse = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: `You are a planning agent. Given a goal, create a numbered action plan.
          Format your response as JSON: 
          {"steps": [{"step": 1, "action": "...", "tool": "...", "notes": "..."}]}`
      },
      { role: "user", content: `Create an action plan for: ${goal}` }
    ]
  });
  
  const plan = JSON.parse(planResponse.choices[0].message.content);
  console.log("Generated Plan:");
  plan.steps.forEach(s => console.log(`  ${s.step}. ${s.action} [Tool: ${s.tool}]`));
  
  // Step 2: Execute the plan step by step
  const results = [];
  for (const step of plan.steps) {
    console.log(`\nExecuting step ${step.step}: ${step.action}`);
    
    const executionResponse = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: "Execute this step of the plan and describe what you did and the result."
        },
        {
          role: "user",
          content: `Overall goal: ${goal}\nCurrent step: ${step.action}\nNotes: ${step.notes}`
        }
      ]
    });
    
    const result = executionResponse.choices[0].message.content;
    results.push({ step: step.step, action: step.action, result });
    console.log(`  Result: ${result.substring(0, 100)}...`);
  }
  
  // Step 3: Synthesize final summary
  const summaryResponse = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "Summarize what was accomplished in a concise report." },
      { role: "user", content: `Goal: ${goal}\n\nResults: ${JSON.stringify(results)}` }
    ]
  });
  
  console.log("\n=== Final Summary ===");
  console.log(summaryResponse.choices[0].message.content);
}

planningAgent("Research the top 3 trends in AI agents and write a brief report");
```

**Key Takeaways:**
- Planning converts a high-level goal into executable steps
- Plans should be adaptive — update when something fails
- Use planning for multi-step, complex tasks where order matters
- Tools like Google Deep Research use planning extensively (searches → analyzes → refines → reports)

---

### Chapter 7: Multi-Agent Collaboration

**The simple idea:** Instead of one all-powerful agent trying to do everything, build a *team* of specialized agents — like a company where different departments handle different things.

**Real-world analogy:** A film production involves a director, actors, cinematographers, editors, and sound engineers. Each expert does their part; together they create something none could alone.

![Multi-Agent](agentic_images/multi_agent.png)

**Collaboration patterns:**
- **Sequential Handoff** — Agent A finishes, passes work to Agent B
- **Parallel Processing** — Multiple agents work on different parts simultaneously
- **Hierarchical** — A manager agent delegates to worker agents
- **Debate & Consensus** — Multiple agents argue different positions to reach a better answer
- **Critic-Reviewer** — One agent creates, another audits it

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Define specialized agents
async function researcherAgent(topic) {
  console.log(`[RESEARCHER] Researching: ${topic}`);
  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: "You are a Senior Research Analyst. Find the top 3 trends and key data points on the topic."
      },
      { role: "user", content: topic }
    ]
  });
  return response.choices[0].message.content;
}

async function writerAgent(researchFindings, audience) {
  console.log(`[WRITER] Writing blog post from research...`);
  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `You are a Technical Content Writer. Write an engaging 400-word blog post for: ${audience}. 
          Use simple language. Use the research findings provided.`
      },
      { role: "user", content: `Research Findings:\n${researchFindings}` }
    ]
  });
  return response.choices[0].message.content;
}

async function editorAgent(draftPost) {
  console.log(`[EDITOR] Reviewing and improving the draft...`);
  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `You are a Chief Editor. Review the blog post and:
          1. Fix grammatical errors
          2. Improve clarity and flow
          3. Ensure it's engaging
          4. Add a compelling headline if missing
          Return the improved version only.`
      },
      { role: "user", content: draftPost }
    ]
  });
  return response.choices[0].message.content;
}

// Orchestrate the multi-agent team (sequential handoffs)
async function contentCreationTeam(topic, targetAudience) {
  console.log(`\n🚀 Starting multi-agent content creation team\n`);
  
  // Agent 1: Research
  const research = await researcherAgent(topic);
  
  // Agent 2: Write (uses research output)
  const draft = await writerAgent(research, targetAudience);
  
  // Agent 3: Edit (uses draft output)
  const finalPost = await editorAgent(draft);
  
  console.log("\n=== FINAL BLOG POST ===\n");
  console.log(finalPost);
  return finalPost;
}

contentCreationTeam(
  "The impact of AI agents on software development in 2025",
  "developers with 2-3 years of experience"
);
```

**Key Takeaways:**
- Specialized agents outperform generalist agents on complex tasks
- Sequential handoffs are easiest to implement
- Use hierarchical patterns (manager + workers) for complex workflows
- Communication between agents needs clear formats (JSON, structured text)

---

## Part 2 — Memory & Adaptation

---

### Chapter 8: Memory Management

**The simple idea:** Without memory, every conversation with an agent starts from zero. Memory lets agents remember you across sessions, track ongoing tasks, and learn from past interactions.

**Two types of memory:**

| Type | What It Is | Example |
|------|-----------|---------|
| **Short-term (context window)** | What the AI remembers *within a conversation* | The messages sent so far |
| **Long-term (external storage)** | What persists *across conversations* | User preferences, past summaries |

![Memory Management](agentic_images/memory.png)

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

class AgentWithMemory {
  constructor(userId) {
    this.userId = userId;
    this.conversationHistory = []; // Short-term memory (current session)
    this.longTermMemory = {};       // Long-term memory (persists across sessions)
    
    // System prompt that gives the agent its role
    this.systemPrompt = {
      role: "system",
      content: "You are a helpful personal assistant. Remember user preferences and context."
    };
  }
  
  // Save important facts to long-term memory
  saveToLongTermMemory(key, value) {
    this.longTermMemory[key] = value;
    console.log(`[MEMORY] Saved: ${key} = ${value}`);
  }
  
  // Retrieve context from long-term memory
  getLongTermContext() {
    if (Object.keys(this.longTermMemory).length === 0) return "";
    return "\n\nUser context from past sessions:\n" + 
      Object.entries(this.longTermMemory)
        .map(([k, v]) => `- ${k}: ${v}`)
        .join("\n");
  }
  
  async chat(userMessage) {
    // Add user message to short-term memory
    this.conversationHistory.push({ role: "user", content: userMessage });
    
    // Build messages: system prompt + long-term context + conversation history
    const systemWithContext = {
      role: "system",
      content: this.systemPrompt.content + this.getLongTermContext()
    };
    
    const messages = [systemWithContext, ...this.conversationHistory];
    
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages
    });
    
    const assistantMessage = response.choices[0].message.content;
    
    // Add response to short-term memory
    this.conversationHistory.push({ role: "assistant", content: assistantMessage });
    
    // Auto-extract and save important facts to long-term memory
    if (userMessage.toLowerCase().includes("my name is")) {
      const name = userMessage.match(/my name is (\w+)/i)?.[1];
      if (name) this.saveToLongTermMemory("user_name", name);
    }
    if (userMessage.toLowerCase().includes("i prefer") || userMessage.toLowerCase().includes("i like")) {
      this.saveToLongTermMemory("preference_" + Date.now(), userMessage);
    }
    
    return assistantMessage;
  }
  
  // Summarize conversation to reduce token count (context compression)
  async compressHistory() {
    if (this.conversationHistory.length < 10) return;
    
    const summaryResponse = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "Summarize this conversation in bullet points, preserving key facts." },
        { role: "user", content: this.conversationHistory.map(m => `${m.role}: ${m.content}`).join("\n") }
      ]
    });
    
    const summary = summaryResponse.choices[0].message.content;
    this.saveToLongTermMemory("conversation_summary", summary);
    this.conversationHistory = []; // Clear short-term memory
    console.log("[MEMORY] History compressed and saved");
  }
}

// Test the memory system
async function testMemoryAgent() {
  const agent = new AgentWithMemory("user123");
  
  let reply = await agent.chat("Hi! My name is Alex and I prefer concise answers.");
  console.log("Agent:", reply);
  
  reply = await agent.chat("What's the weather like today?");
  console.log("Agent:", reply);
  
  reply = await agent.chat("What's my name?"); // Agent should remember!
  console.log("Agent:", reply);
}

testMemoryAgent();
```

**Key Takeaways:**
- Short-term memory = the current chat; long-term = database/vector store
- Without memory, agents can't maintain relationships or track ongoing work
- Compress old conversations to save token costs
- Use vector databases for semantic memory (finding relevant past info)

---

### Chapter 9: Learning and Adaptation

**The simple idea:** An agent that can't learn from experience will always make the same mistakes. Learning agents improve over time — they update their knowledge, refine their strategies, and adapt to new situations automatically.

**Learning methods agents use:**
- **Reinforcement Learning** — Gets rewards for good actions, penalties for bad ones
- **Supervised Learning** — Learns from labeled examples
- **Few-shot Learning** — Adapts to new tasks with just a few examples
- **Self-improvement** — Actually modifies its own code/prompts (advanced!)

**Remarkable example — SICA (Self-Improving Coding Agent):**

SICA is an agent that *rewrites its own code* to become better at programming tasks. It:
1. Reviews its past performance
2. Identifies weaknesses
3. Writes new tools to address those weaknesses
4. Tests itself again
5. Keeps whichever version performs better

This is a real system that created better code editors, search tools, and analyzers — all by itself.

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

class AdaptiveLearningAgent {
  constructor() {
    this.episodicMemory = []; // Records of past interactions and outcomes
    this.successfulPatterns = {}; // Patterns that worked well
    this.failedPatterns = {};    // Patterns to avoid
  }
  
  // Record what happened and whether it was successful
  recordExperience(input, output, wasSuccessful, feedback) {
    const experience = { input, output, wasSuccessful, feedback, timestamp: Date.now() };
    this.episodicMemory.push(experience);
    
    // Learn: track patterns that work vs don't work
    const pattern = input.substring(0, 50); // Simplified pattern key
    if (wasSuccessful) {
      this.successfulPatterns[pattern] = (this.successfulPatterns[pattern] || 0) + 1;
    } else {
      this.failedPatterns[pattern] = (this.failedPatterns[pattern] || 0) + 1;
    }
  }
  
  // Get lessons learned from past experiences
  getLearnedInsights() {
    const successes = Object.entries(this.successfulPatterns)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([pattern, count]) => `"${pattern}" (worked ${count} times)`);
    
    const failures = Object.entries(this.failedPatterns)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([pattern, count]) => `"${pattern}" (failed ${count} times)`);
    
    return { successes, failures };
  }
  
  // Respond with awareness of past learning
  async respondWithLearning(userMessage) {
    const insights = this.getLearnedInsights();
    
    const systemPrompt = `You are an adaptive learning assistant.
      
      Based on past experience:
      - Successful approaches: ${insights.successes.join(", ") || "none yet"}
      - Approaches to avoid: ${insights.failures.join(", ") || "none yet"}
      
      Apply these lessons to give a better response.`;
    
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage }
      ]
    });
    
    return response.choices[0].message.content;
  }
  
  // Simulate the agent improving its own system prompt based on feedback
  async refineOwnInstructions(currentInstructions, recentFeedback) {
    const response = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "You improve AI system prompts. Refine the given prompt based on the feedback provided."
        },
        {
          role: "user",
          content: `Current instructions:\n${currentInstructions}\n\nRecent feedback:\n${recentFeedback}\n\nImproved instructions:`
        }
      ]
    });
    
    return response.choices[0].message.content;
  }
}

// Demo usage
async function testLearningAgent() {
  const agent = new AdaptiveLearningAgent();
  
  // Simulate some past experiences
  agent.recordExperience("explain technical concept", "gave too much detail", false, "too complex");
  agent.recordExperience("explain technical concept", "used simple analogy", true, "perfect!");
  agent.recordExperience("write code", "wrote clean, commented code", true, "excellent!");
  
  // Now respond using learned experience
  const response = await agent.respondWithLearning("Explain what an API is");
  console.log("Response with learning:", response);
  
  // Demonstrate self-improvement
  const improvedInstructions = await agent.refineOwnInstructions(
    "You are a helpful assistant.",
    "Users say responses are too technical and long. They want shorter, simpler answers with analogies."
  );
  console.log("\nImproved instructions:", improvedInstructions);
}

testLearningAgent();
```

**Key Takeaways:**
- Agents should record what worked and what didn't
- Episodic memory (remembering past events) helps avoid repeating mistakes
- Self-improvement is the frontier — agents that modify their own prompts or code
- AlphaEvolve (Google) uses this to discover new mathematical algorithms

---

### Chapter 10: Model Context Protocol (MCP)

**The simple idea:** MCP is like a **universal plug standard for AI agents**. Instead of custom connections between each AI and each tool, MCP creates one standard interface so any AI can connect to any tool.

**Real-world analogy:** Before USB, every device had its own connector. USB standardized everything — now you can connect any device to any computer. MCP does the same for AI tools.

**MCP vs. Regular Tool Calling:**

| Feature | Tool Function Calling | MCP |
|---------|----------------------|-----|
| Standard | Varies by provider | Open standard |
| Discovery | Static (you tell it what's available) | Dynamic (AI discovers tools at runtime) |
| Reusability | Tightly coupled | Reusable across different AIs |
| Architecture | One-to-one | Client-server |

**JavaScript Example (MCP Server with FastMCP concept):**

```javascript
// Simulating an MCP-style tool server
const http = require("http");

// MCP Server: Exposes tools that any AI can discover and call
const mcpServer = {
  // Tool catalog — this is what the AI "discovers"
  tools: [
    {
      name: "greet",
      description: "Generates a personalized greeting",
      parameters: {
        type: "object",
        properties: {
          name: { type: "string", description: "The person's name" }
        },
        required: ["name"]
      }
    },
    {
      name: "calculate",
      description: "Performs a mathematical calculation",
      parameters: {
        type: "object",
        properties: {
          expression: { type: "string", description: "Math expression to evaluate, e.g. '2 + 2'" }
        },
        required: ["expression"]
      }
    }
  ],
  
  // Tool execution
  execute: async (toolName, args) => {
    switch (toolName) {
      case "greet":
        return { result: `Hello, ${args.name}! Nice to meet you.` };
      case "calculate":
        try {
          // Safe eval for math only
          const result = Function(`"use strict"; return (${args.expression})`)();
          return { result: `${args.expression} = ${result}` };
        } catch (e) {
          return { error: "Invalid expression" };
        }
      default:
        return { error: `Unknown tool: ${toolName}` };
    }
  }
};

// MCP Client: How an AI agent would use the MCP server
class MCPClient {
  constructor(serverTools) {
    this.availableTools = serverTools;
  }
  
  async discoverTools() {
    // In real MCP, this is a network call to /.well-known/mcp
    console.log("Discovered tools:", this.availableTools.map(t => t.name));
    return this.availableTools;
  }
  
  async callTool(toolName, args) {
    console.log(`Calling MCP tool: ${toolName} with`, args);
    return await mcpServer.execute(toolName, args);
  }
}

// Agent that uses MCP
const { OpenAI } = require("openai");
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function agentWithMCP(userMessage) {
  const mcpClient = new MCPClient(mcpServer.tools);
  const availableTools = await mcpClient.discoverTools();
  
  // Convert MCP tool format to OpenAI tool format
  const openAITools = availableTools.map(tool => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters
    }
  }));
  
  // Ask AI which tool to use
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: userMessage }],
    tools: openAITools,
    tool_choice: "auto"
  });
  
  const message = response.choices[0].message;
  
  if (message.tool_calls) {
    for (const call of message.tool_calls) {
      const toolResult = await mcpClient.callTool(
        call.function.name,
        JSON.parse(call.function.arguments)
      );
      console.log("Tool result:", toolResult);
    }
  }
}

agentWithMCP("Greet my friend Sarah and also calculate 15 * 7");
```

**Key Takeaways:**
- MCP standardizes how AI talks to external tools
- Enables dynamic tool discovery — AI finds out what tools exist at runtime
- One MCP server can serve many different AI systems
- Use MCP for enterprise scale; use direct tool calling for simpler projects

---

### Chapter 11: Goal Setting and Monitoring

**The simple idea:** Agents need clear goals and a way to check if they're making progress. Without monitoring, an agent might wander off course or never know if it succeeded.

**SMART Goals for AI:** Goals should be Specific, Measurable, Achievable, Relevant, and Time-bound.

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

class GoalDrivenAgent {
  constructor(goal, successCriteria) {
    this.goal = goal;
    this.successCriteria = successCriteria;
    this.iterationCount = 0;
    this.maxIterations = 5;
    this.currentOutput = null;
  }
  
  // Generate or improve output
  async produce(previousOutput, previousFeedback) {
    const prompt = previousOutput
      ? `Improve this based on feedback:\nPrevious: ${previousOutput}\nFeedback: ${previousFeedback}`
      : `Produce output for: ${this.goal}`;
    
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "You are a focused assistant. Produce exactly what is asked." },
        { role: "user", content: prompt }
      ]
    });
    
    return response.choices[0].message.content;
  }
  
  // Monitor: check if the goal is achieved
  async checkGoalAchieved(output) {
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a goal evaluator. Check if the output meets ALL these criteria:
            ${this.successCriteria.map((c, i) => `${i + 1}. ${c}`).join("\n")}
            
            Reply with ONLY: "GOAL_ACHIEVED" or a bullet list of what's missing.`
        },
        {
          role: "user",
          content: `Goal: ${this.goal}\n\nOutput to evaluate:\n${output}`
        }
      ]
    });
    
    return response.choices[0].message.content;
  }
  
  // Run the goal-driven loop
  async run() {
    console.log(`\n🎯 Goal: ${this.goal}`);
    console.log(`📋 Success criteria: ${this.successCriteria.join(", ")}\n`);
    
    let feedback = null;
    
    while (this.iterationCount < this.maxIterations) {
      this.iterationCount++;
      console.log(`\n--- Iteration ${this.iterationCount} ---`);
      
      // Produce or improve output
      this.currentOutput = await this.produce(this.currentOutput, feedback);
      console.log("Output (preview):", this.currentOutput.substring(0, 150) + "...");
      
      // Monitor progress
      const evaluation = await this.checkGoalAchieved(this.currentOutput);
      
      if (evaluation.includes("GOAL_ACHIEVED")) {
        console.log(`\n✅ Goal achieved in ${this.iterationCount} iteration(s)!`);
        return this.currentOutput;
      }
      
      feedback = evaluation;
      console.log("Feedback:", feedback);
    }
    
    console.log(`\n⚠️ Max iterations reached. Best output so far returned.`);
    return this.currentOutput;
  }
}

// Test
const agent = new GoalDrivenAgent(
  "Write a JavaScript function that reverses a string",
  [
    "Must handle empty strings",
    "Must handle null input without crashing",
    "Must include JSDoc comments",
    "Must have at least one example in comments"
  ]
);

agent.run().then(output => {
  console.log("\n=== FINAL OUTPUT ===");
  console.log(output);
});
```

**Key Takeaways:**
- Define goals *before* the agent starts working
- Monitor progress continuously, not just at the end
- Feedback loops let agents course-correct automatically
- Goal monitoring + Reflection = a self-improving system

---

## Part 3 — Robustness & Safety

---

### Chapter 12: Exception Handling and Recovery

**The simple idea:** Things go wrong. APIs fail. Tools crash. Data is corrupted. A resilient agent doesn't just crash — it detects the problem, tries to recover, and degrades gracefully.

**Key strategies:**
- **Retry** — Try again (maybe with different parameters)
- **Fallback** — Use a backup tool/model if the primary fails
- **Graceful degradation** — Do less, but don't fail completely
- **Escalation** — Alert a human when the agent can't fix it

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Utility: retry with exponential backoff
async function withRetry(fn, maxRetries = 3, baseDelay = 1000) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (attempt === maxRetries) throw error;
      const delay = baseDelay * Math.pow(2, attempt - 1);
      console.log(`[RETRY] Attempt ${attempt} failed. Retrying in ${delay}ms... Error: ${error.message}`);
      await new Promise(r => setTimeout(r, delay));
    }
  }
}

// Primary tool (might fail)
async function getPreciseLocationInfo(address) {
  // Simulating an API call that sometimes fails
  if (Math.random() < 0.5) throw new Error("Location API unavailable");
  return { address, lat: 51.5074, lng: -0.1278, type: "precise" };
}

// Fallback tool (more reliable, less precise)
async function getGeneralAreaInfo(city) {
  return { city, type: "general", approximate: true };
}

// Exception-aware agent
class ResilientAgent {
  constructor() {
    this.errors = [];
  }
  
  logError(step, error) {
    this.errors.push({ step, error: error.message, timestamp: Date.now() });
    console.error(`[ERROR] ${step}: ${error.message}`);
  }
  
  async getLocationWithFallback(address) {
    // Try primary (precise) tool first
    try {
      const result = await withRetry(() => getPreciseLocationInfo(address));
      console.log("[SUCCESS] Got precise location:", result);
      return result;
    } catch (primaryError) {
      this.logError("precise_location", primaryError);
      
      // Fallback: extract city and try general lookup
      console.log("[FALLBACK] Falling back to general area info...");
      try {
        const city = address.split(",")[0]; // Simple city extraction
        const result = await getGeneralAreaInfo(city);
        console.log("[FALLBACK SUCCESS] Got general location:", result);
        return result;
      } catch (fallbackError) {
        this.logError("general_location", fallbackError);
        
        // Final fallback: return a minimal response
        console.log("[GRACEFUL DEGRADATION] Returning minimal response");
        return { error: "Could not retrieve location", address, status: "degraded" };
      }
    }
  }
  
  // Generate final response even if tools partially failed
  async respond(userQuery) {
    let locationData;
    
    try {
      locationData = await this.getLocationWithFallback(userQuery);
    } catch (error) {
      locationData = { error: "Location services unavailable" };
    }
    
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: "Answer the user's location query. If location data is degraded or has errors, acknowledge this honestly."
        },
        {
          role: "user",
          content: `User query: ${userQuery}\nLocation data retrieved: ${JSON.stringify(locationData)}`
        }
      ]
    });
    
    return response.choices[0].message.content;
  }
  
  getErrorReport() {
    return this.errors.length > 0
      ? `Encountered ${this.errors.length} error(s) during execution: ${this.errors.map(e => e.step).join(", ")}`
      : "No errors encountered";
  }
}

async function testResilienceAgent() {
  const agent = new ResilientAgent();
  const response = await agent.respond("London, UK");
  console.log("\nFinal Response:", response);
  console.log("\nError Report:", agent.getErrorReport());
}

testResilienceAgent();
```

**Key Takeaways:**
- Always retry transient errors with exponential backoff
- Have fallback tools/models ready when primary ones fail
- Log all errors — you need this for debugging
- Graceful degradation means partial success beats complete failure

---

### Chapter 13: Human-in-the-Loop

**The simple idea:** Some decisions are too important, too ambiguous, or too risky for an AI to make alone. Human-in-the-Loop (HITL) keeps humans involved at critical decision points.

**When to use HITL:**
- High-stakes decisions (financial, medical, legal)
- Ambiguous situations requiring judgment
- Content moderation edge cases
- When errors could cause serious harm

**HITL Variants:**
- **Human-in-the-loop** — Human approves each critical action
- **Human-on-the-loop** — Human sets policy; AI acts automatically within that policy; human monitors
- **Human-as-the-loop** — Human provides all feedback for AI training (like data labeling)

**JavaScript Example:**

```javascript
const readline = require("readline");
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Simulate human approval for high-stakes actions
function askHumanApproval(action, details) {
  return new Promise((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    console.log(`\n⚠️  HUMAN REVIEW REQUIRED`);
    console.log(`Action: ${action}`);
    console.log(`Details: ${details}`);
    rl.question(`Approve? (yes/no): `, (answer) => {
      rl.close();
      resolve(answer.toLowerCase() === "yes" || answer.toLowerCase() === "y");
    });
  });
}

class HumanInLoopAgent {
  constructor() {
    this.actionLog = [];
  }
  
  // Actions that require human approval
  async performHighStakesAction(actionType, params, autoApprove = false) {
    const details = JSON.stringify(params);
    
    let approved;
    if (autoApprove) {
      // Simulated approval for demos
      approved = true;
      console.log(`[AUTO-APPROVED] ${actionType}: ${details}`);
    } else {
      approved = await askHumanApproval(actionType, details);
    }
    
    if (approved) {
      this.actionLog.push({ action: actionType, params, approved: true, timestamp: Date.now() });
      console.log(`✅ Action approved and executed: ${actionType}`);
      return { success: true, action: actionType, params };
    } else {
      this.actionLog.push({ action: actionType, params, approved: false, timestamp: Date.now() });
      console.log(`❌ Action rejected by human: ${actionType}`);
      return { success: false, reason: "Human rejected the action" };
    }
  }
  
  // Escalation: when AI can't handle something
  async escalate(issue, context) {
    console.log(`\n🚨 ESCALATING TO HUMAN SUPPORT`);
    console.log(`Issue: ${issue}`);
    console.log(`Context: ${context}`);
    // In real system, create a support ticket, send Slack message, etc.
    return "Escalated to human support team. Ticket #12345 created.";
  }
  
  // Main agent logic with HITL checkpoints
  async handleRequest(userRequest) {
    // Step 1: Classify the request
    const classificationResponse = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `Classify the request as:
            - "safe" (informational, no action needed)
            - "low_risk" (simple action, auto-approve OK)
            - "high_risk" (needs human approval before action)
            - "escalate" (too complex for AI, needs human)
            Reply with ONLY the classification and a brief reason: {"classification": "...", "reason": "..."}`
        },
        { role: "user", content: userRequest }
      ]
    });
    
    const classification = JSON.parse(classificationResponse.choices[0].message.content);
    console.log(`\nRequest classified as: ${classification.classification} — ${classification.reason}`);
    
    // Route based on risk level
    switch (classification.classification) {
      case "safe":
        const safeResponse = await client.chat.completions.create({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: userRequest }]
        });
        return safeResponse.choices[0].message.content;
      
      case "low_risk":
        const result = await this.performHighStakesAction("low_risk_action", { request: userRequest }, true);
        return result.success ? `Completed: ${userRequest}` : `Could not complete: ${result.reason}`;
      
      case "high_risk":
        const highRiskResult = await this.performHighStakesAction(
          "high_risk_action", 
          { request: userRequest },
          false // REQUIRES ACTUAL HUMAN APPROVAL
        );
        return highRiskResult.success
          ? `Action completed after human approval`
          : `Action rejected by human oversight`;
      
      case "escalate":
        return await this.escalate("Complex request requiring human judgment", userRequest);
    }
  }
}

async function testHITLAgent() {
  const agent = new HumanInLoopAgent();
  
  // Simulate (with auto-approve for demo)
  const requests = [
    "What is the capital of France?",          // safe
    "Send a welcome email to new@user.com",    // low_risk
  ];
  
  for (const req of requests) {
    console.log(`\nUser: ${req}`);
    const response = await agent.handleRequest(req);
    console.log(`Agent: ${response}`);
  }
}

testHITLAgent();
```

**Key Takeaways:**
- HITL doesn't mean humans do everything — it means humans approve critical decisions
- Human-on-the-loop is a good middle ground: humans set policy, AI executes
- Escalation policies define *when* the AI must hand off to humans
- HITL is essential in finance, healthcare, legal, and autonomous systems

---

### Chapter 14: Knowledge Retrieval (RAG)

**The simple idea:** An AI's training data has a cutoff date and doesn't include your private documents. RAG lets the AI look things up — from your own documents, databases, or the web — before answering, giving it up-to-date, specific knowledge.

**How RAG works:**

1. Your documents are split into chunks and converted to **embeddings** (numerical representations of meaning)
2. When you ask a question, your question is also converted to an embedding
3. The system finds the most *semantically similar* document chunks to your question
4. Those chunks are sent to the AI along with your question
5. The AI uses the retrieved context to give an accurate, grounded answer

![RAG](agentic_images/rag.png)

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Simple in-memory vector store (production would use Pinecone, Weaviate, etc.)
class SimpleVectorStore {
  constructor() {
    this.documents = [];
    this.embeddings = [];
  }
  
  // Calculate cosine similarity between two vectors
  cosineSimilarity(vecA, vecB) {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }
  
  // Add a document with its embedding
  async addDocument(text, metadata = {}) {
    const embeddingResponse = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: text
    });
    const embedding = embeddingResponse.data[0].embedding;
    this.documents.push({ text, metadata });
    this.embeddings.push(embedding);
    console.log(`Added document: "${text.substring(0, 60)}..."`);
  }
  
  // Find top-k most similar documents to a query
  async search(query, topK = 3) {
    const queryEmbeddingResponse = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: query
    });
    const queryEmbedding = queryEmbeddingResponse.data[0].embedding;
    
    // Calculate similarity scores
    const similarities = this.embeddings.map((embedding, index) => ({
      index,
      score: this.cosineSimilarity(queryEmbedding, embedding)
    }));
    
    // Sort by similarity and return top-k
    const topResults = similarities
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map(r => ({ ...this.documents[r.index], score: r.score }));
    
    return topResults;
  }
}

class RAGAgent {
  constructor() {
    this.vectorStore = new SimpleVectorStore();
  }
  
  // Load documents into the knowledge base
  async indexDocuments(documents) {
    console.log(`\nIndexing ${documents.length} documents...`);
    for (const doc of documents) {
      await this.vectorStore.addDocument(doc.text, doc.metadata);
    }
    console.log("Indexing complete!\n");
  }
  
  // Answer a question using RAG
  async answer(question) {
    console.log(`\nQuestion: ${question}`);
    
    // Step 1: Retrieve relevant context
    const relevantDocs = await this.vectorStore.search(question, 3);
    console.log(`Retrieved ${relevantDocs.length} relevant documents`);
    
    // Step 2: Build context from retrieved docs
    const context = relevantDocs
      .map((doc, i) => `[Source ${i + 1}] ${doc.text}`)
      .join("\n\n");
    
    // Step 3: Generate answer using retrieved context
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a helpful assistant. Answer questions using ONLY the provided context.
            If the context doesn't contain the answer, say so — don't make things up.
            Always cite which source you're using.`
        },
        {
          role: "user",
          content: `Context:\n${context}\n\nQuestion: ${question}`
        }
      ]
    });
    
    const answer = response.choices[0].message.content;
    console.log("Answer:", answer);
    return answer;
  }
}

// Demo
async function testRAG() {
  const agent = new RAGAgent();
  
  // Index your knowledge base
  await agent.indexDocuments([
    { 
      text: "Our return policy allows returns within 30 days of purchase. Items must be in original condition. Refunds are processed within 5-7 business days.",
      metadata: { source: "return_policy.pdf" }
    },
    {
      text: "Free shipping is available on all orders over $50. Standard shipping takes 3-5 business days. Express shipping (1-2 days) costs $15.",
      metadata: { source: "shipping_policy.pdf" }
    },
    {
      text: "We offer a 1-year warranty on all electronics. The warranty covers manufacturing defects but not physical damage or water damage.",
      metadata: { source: "warranty_terms.pdf" }
    }
  ]);
  
  // Ask questions
  await agent.answer("How long do I have to return an item?");
  await agent.answer("Is shipping free?");
  await agent.answer("What does the warranty cover?");
}

testRAG();
```

**Key Takeaways:**
- RAG gives LLMs access to current, private, or specific knowledge
- Embeddings let you do meaning-based search (not just keyword matching)
- Always cite sources so answers are verifiable
- Advanced: Agentic RAG adds a reasoning layer to evaluate and reconcile retrieved info

---

## Part 4 — Advanced Patterns

---

### Chapter 15: Inter-Agent Communication (A2A)

**The simple idea:** Just like MCP standardizes AI-to-tool connections, A2A (Agent-to-Agent) standardizes how *agents talk to each other* — even if they're built with different frameworks.

**Core concepts:**
- **Agent Card** — Like a business card in JSON, describing what an agent can do
- **Tasks** — The units of work agents pass to each other
- **Artifacts** — The outputs agents produce and share

**JavaScript Example (A2A-style communication):**

```javascript
const express = require("express");
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// A2A Weather Agent (acts as an MCP/A2A server)
class WeatherAgent {
  constructor() {
    this.agentCard = {
      name: "WeatherBot",
      description: "Provides weather information for any city",
      version: "1.0.0",
      skills: [
        {
          id: "get_weather",
          name: "Get Current Weather",
          description: "Returns weather for a given city"
        }
      ]
    };
  }
  
  async handleTask(task) {
    const city = task.message;
    
    // Simulate weather lookup
    const weatherData = {
      london: { temp: "15°C", condition: "Cloudy", humidity: "80%" },
      paris: { temp: "20°C", condition: "Sunny", humidity: "55%" },
      tokyo: { temp: "28°C", condition: "Hot", humidity: "70%" }
    }[city.toLowerCase()] || { temp: "Unknown", condition: "No data" };
    
    return {
      taskId: task.id,
      status: "completed",
      artifact: {
        type: "weather_report",
        data: { city, ...weatherData }
      }
    };
  }
  
  getAgentCard() {
    return this.agentCard;
  }
}

// A2A Client: A coordinator agent that uses other agents
class CoordinatorAgent {
  constructor() {
    this.registeredAgents = {};
  }
  
  // Register an agent (discover via A2A in real systems)
  registerAgent(name, agent) {
    this.registeredAgents[name] = agent;
    console.log(`Registered agent: ${name}`, agent.getAgentCard().skills.map(s => s.id));
  }
  
  // Send a task to a remote agent
  async sendTask(agentName, message) {
    const agent = this.registeredAgents[agentName];
    if (!agent) throw new Error(`Agent not found: ${agentName}`);
    
    const task = {
      id: `task-${Date.now()}`,
      message,
      timestamp: new Date().toISOString()
    };
    
    console.log(`\n[A2A] Sending task to ${agentName}:`, task);
    const result = await agent.handleTask(task);
    console.log(`[A2A] Received result from ${agentName}:`, result);
    return result;
  }
  
  // Coordinate multi-agent workflow
  async planTrip(destination) {
    console.log(`\n🗺️ Coordinating trip planning for: ${destination}`);
    
    // Step 1: Get weather from WeatherBot
    const weatherResult = await this.sendTask("WeatherBot", destination);
    const weather = weatherResult.artifact.data;
    
    // Step 2: Generate trip recommendations using the gathered data
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: "You are a travel planner. Create a brief trip recommendation."
        },
        {
          role: "user",
          content: `Destination: ${destination}
            Current Weather: ${weather.temp}, ${weather.condition}
            Please provide 3 activity recommendations based on this weather.`
        }
      ]
    });
    
    const recommendations = response.choices[0].message.content;
    
    return {
      destination,
      weather,
      recommendations
    };
  }
}

// Demo
async function testA2A() {
  const coordinator = new CoordinatorAgent();
  const weatherBot = new WeatherAgent();
  
  // Register agents (in real A2A, they self-announce via Agent Cards)
  coordinator.registerAgent("WeatherBot", weatherBot);
  
  // Coordinate a trip planning task
  const tripPlan = await coordinator.planTrip("Tokyo");
  
  console.log("\n=== TRIP PLAN ===");
  console.log(`Destination: ${tripPlan.destination}`);
  console.log(`Weather: ${tripPlan.weather.temp}, ${tripPlan.weather.condition}`);
  console.log(`Recommendations:\n${tripPlan.recommendations}`);
}

testA2A();
```

**Key Takeaways:**
- A2A lets agents from different frameworks collaborate
- Agent Cards are how agents announce their capabilities
- Supported by Google, Microsoft, Salesforce, and many others
- A2A complements MCP: MCP = AI-to-tool, A2A = AI-to-AI

---

### Chapter 16: Resource-Aware Optimization

**The simple idea:** Not every question needs the most powerful (and expensive) AI model. Smart agents pick the right tool for the right job — use cheap/fast models for simple questions, powerful models only when needed.

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

class ResourceAwareAgent {
  constructor() {
    this.costTracker = { totalCalls: 0, cheapCalls: 0, expensiveCalls: 0 };
  }
  
  // Classify query complexity (uses cheapest model to decide)
  async classifyQuery(query) {
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini", // Cheapest model for classification
      messages: [
        {
          role: "system",
          content: `Classify query difficulty: "simple" (factual, 1-step), "moderate" (multi-step), "complex" (reasoning-heavy).
            Reply with ONLY one word.`
        },
        { role: "user", content: query }
      ]
    });
    return response.choices[0].message.content.trim().toLowerCase();
  }
  
  // Route to appropriate model based on complexity
  async answer(query) {
    this.costTracker.totalCalls++;
    
    // Step 1: Classify complexity
    const complexity = await this.classifyQuery(query);
    console.log(`Query complexity: ${complexity}`);
    
    let model;
    let modelLabel;
    
    switch (complexity) {
      case "simple":
        model = "gpt-4o-mini"; // Cheapest
        modelLabel = "Fast (cheap)";
        this.costTracker.cheapCalls++;
        break;
      case "moderate":
        model = "gpt-4o-mini"; // Still fast for moderate
        modelLabel = "Balanced";
        this.costTracker.cheapCalls++;
        break;
      case "complex":
        model = "gpt-4o"; // Best model for hard problems
        modelLabel = "Powerful (expensive)";
        this.costTracker.expensiveCalls++;
        break;
      default:
        model = "gpt-4o-mini";
        modelLabel = "Default";
    }
    
    console.log(`Using model: ${modelLabel} (${model})`);
    
    // Step 2: Generate answer with selected model
    const response = await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: query }]
    });
    
    return {
      answer: response.choices[0].message.content,
      model,
      complexity
    };
  }
  
  getCostReport() {
    const savingsPercent = ((this.costTracker.cheapCalls / this.costTracker.totalCalls) * 100).toFixed(1);
    return `Total calls: ${this.costTracker.totalCalls} | Cheap: ${this.costTracker.cheapCalls} | Expensive: ${this.costTracker.expensiveCalls} | Cheap rate: ${savingsPercent}%`;
  }
}

// Test with different complexities
async function testResourceAgent() {
  const agent = new ResourceAwareAgent();
  
  const queries = [
    "What is 2 + 2?",                                          // simple
    "What are the capitals of France and Germany?",            // simple
    "Explain quantum entanglement in simple terms",            // moderate
    "Write a detailed analysis of how AI will affect the economy in the next 10 years" // complex
  ];
  
  for (const query of queries) {
    console.log(`\nQuery: "${query}"`);
    const result = await agent.answer(query);
    console.log(`Answer (${result.complexity}/${result.model}):`, result.answer.substring(0, 100) + "...");
  }
  
  console.log("\n📊 Cost Report:", agent.getCostReport());
}

testResourceAgent();
```

**Key Takeaways:**
- Classify query complexity before picking a model
- Simple queries → fast cheap models; complex → powerful expensive models
- Track costs and continuously optimize routing decisions
- Fallback mechanisms ensure availability even when primary models are down

---

### Chapter 17: Reasoning Techniques

**The simple idea:** How an AI *thinks through* a problem greatly affects the quality of its answer. Different reasoning techniques unlock different levels of problem-solving.

**Key techniques:**

**1. Chain-of-Thought (CoT)** — "Think step by step"
The AI writes out its reasoning before answering, like showing your work in math.

**2. Tree-of-Thought (ToT)** — Explore multiple paths
Instead of one line of reasoning, explore branches, evaluate them, pick the best.

**3. ReAct** — Reason then Act
Thought → Action → Observation → repeat. The AI thinks, does something, sees what happened, and adjusts.

**4. Self-Correction** — Review and improve own answers
Generate → Critique → Revise cycle.

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Chain-of-Thought Reasoning
async function chainOfThought(problem) {
  console.log("\n=== Chain-of-Thought Reasoning ===");
  
  const response = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: `You are a careful reasoner. When solving problems:
          1. Break down the problem into sub-problems
          2. Think through each step explicitly  
          3. Show your work before giving the final answer
          4. Check your reasoning before concluding
          
          Format: 
          THINKING: [step by step reasoning]
          ANSWER: [final answer]`
      },
      { role: "user", content: problem }
    ]
  });
  
  console.log(response.choices[0].message.content);
  return response.choices[0].message.content;
}

// ReAct: Reason and Act in a loop
async function reactAgent(goal, availableTools) {
  console.log("\n=== ReAct Agent ===");
  const thoughts = [];
  let currentKnowledge = "";
  
  for (let step = 0; step < 3; step++) {
    // THOUGHT: What should I do next?
    const thoughtResponse = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a ReAct agent. Given a goal and current knowledge, decide what to do next.
            Available tools: ${availableTools.join(", ")}
            Output format: {"thought": "...", "action": "...", "action_input": "..."} or {"thought": "...", "final_answer": "..."}`
        },
        {
          role: "user",
          content: `Goal: ${goal}\nCurrent knowledge: ${currentKnowledge}\nWhat's your next step?`
        }
      ]
    });
    
    const decision = JSON.parse(thoughtResponse.choices[0].message.content);
    console.log(`\nStep ${step + 1}:`);
    console.log(`  THOUGHT: ${decision.thought}`);
    
    if (decision.final_answer) {
      console.log(`  FINAL ANSWER: ${decision.final_answer}`);
      return decision.final_answer;
    }
    
    console.log(`  ACTION: ${decision.action}(${decision.action_input})`);
    
    // OBSERVE: Simulate action result
    const observation = `[Simulated result of ${decision.action} with "${decision.action_input}"]`;
    currentKnowledge += `\n- ${decision.action}: ${observation}`;
    console.log(`  OBSERVATION: ${observation}`);
    thoughts.push(decision);
  }
}

// Test
async function testReasoning() {
  // CoT for a math problem
  await chainOfThought(
    "A store has 120 apples. They sold 35% on Monday and 25% of the remaining on Tuesday. How many apples are left?"
  );
  
  // ReAct for a research task  
  await reactAgent(
    "Find out what the weather is in Paris and calculate how many degrees warmer it is than 10°C",
    ["search_weather", "calculate", "search_web"]
  );
}

testReasoning();
```

**Key Takeaways:**
- "Think step by step" (CoT) dramatically improves accuracy on complex problems
- ReAct lets agents loop through thought → action → observation
- More thinking time = better answers (Scaling Inference Law)
- Use Tree-of-Thought when there are multiple possible solution paths

---

### Chapter 18: Guardrails / Safety Patterns

**The simple idea:** Agents are powerful but can go wrong — giving harmful advice, exposing private data, or taking dangerous actions. Guardrails are the safety systems that keep them in line.

**Types of guardrails:**
- **Input validation** — Check what the user is asking before processing
- **Output filtering** — Review AI responses before showing them
- **Action sandboxing** — Limit what tools the agent can actually run
- **Content moderation** — Filter harmful, biased, or inappropriate content
- **Rate limiting** — Prevent abuse

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

class GuardrailledAgent {
  constructor() {
    this.blockedTopics = ["how to make weapons", "illegal activities", "personal data of others"];
    this.maxResponseLength = 2000;
    this.requestCount = new Map(); // Rate limiting
  }
  
  // Input guardrail: check if request is safe
  async validateInput(userMessage, userId) {
    // Rate limiting: max 10 requests per minute per user
    const now = Date.now();
    const userRequests = this.requestCount.get(userId) || [];
    const recentRequests = userRequests.filter(time => now - time < 60000);
    
    if (recentRequests.length >= 10) {
      return { allowed: false, reason: "Rate limit exceeded. Please wait a minute." };
    }
    
    this.requestCount.set(userId, [...recentRequests, now]);
    
    // Content safety check
    const safetyResponse = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a content safety classifier. Check if this request:
            1. Asks for harmful or dangerous information
            2. Involves illegal activities
            3. Requests private data about others
            
            Reply with ONLY: {"safe": true} or {"safe": false, "reason": "brief explanation"}`
        },
        { role: "user", content: userMessage }
      ]
    });
    
    const safetyCheck = JSON.parse(safetyResponse.choices[0].message.content);
    
    if (!safetyCheck.safe) {
      return { allowed: false, reason: safetyCheck.reason };
    }
    
    return { allowed: true };
  }
  
  // Output guardrail: review response before sending
  async validateOutput(response) {
    // Length check
    if (response.length > this.maxResponseLength) {
      response = response.substring(0, this.maxResponseLength) + "\n\n[Response truncated for safety]";
    }
    
    // PII detection (simplified)
    const piiPatterns = [
      /\b\d{3}-\d{2}-\d{4}\b/g,  // SSN
      /\b\d{16}\b/g,               // Credit card
      /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi // Email (if shouldn't be included)
    ];
    
    for (const pattern of piiPatterns) {
      response = response.replace(pattern, "[REDACTED]");
    }
    
    return response;
  }
  
  // Safe agent response
  async respond(userMessage, userId = "default") {
    // Input check
    const inputCheck = await this.validateInput(userMessage, userId);
    if (!inputCheck.allowed) {
      return `I cannot help with that request. ${inputCheck.reason}`;
    }
    
    // Generate response
    let response;
    try {
      const aiResponse = await client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content: "You are a helpful assistant. Never provide harmful, illegal, or dangerous information. If asked for such, politely decline."
          },
          { role: "user", content: userMessage }
        ]
      });
      response = aiResponse.choices[0].message.content;
    } catch (error) {
      return "I encountered an error. Please try again.";
    }
    
    // Output check
    response = await this.validateOutput(response);
    
    return response;
  }
}

async function testGuardrails() {
  const agent = new GuardrailledAgent();
  
  const testMessages = [
    "What is the capital of Japan?",                    // Safe
    "Can you help me write a poem about autumn?",       // Safe
    "Tell me how to make a dangerous substance",        // Should be blocked
  ];
  
  for (const message of testMessages) {
    console.log(`\nUser: ${message}`);
    const response = await agent.respond(message, "user123");
    console.log(`Agent: ${response.substring(0, 200)}...`);
  }
}

testGuardrails();
```

**Key Takeaways:**
- Every agent needs input and output guardrails
- Rate limiting prevents abuse and runaway costs
- PII (Personally Identifiable Information) must never leak
- Layer your safety: validate input AND review output

---

### Chapter 19: Evaluation and Monitoring

**The simple idea:** You can't improve what you don't measure. Agents need continuous evaluation — tracking accuracy, latency, cost, and quality — especially in production where things change over time.

**Evaluation methods:**

| Method | Best For | Limitation |
|--------|----------|-----------|
| **Exact match** | Simple factual checks | Doesn't handle paraphrasing |
| **LLM-as-a-Judge** | Subjective quality, helpfulness | LLM has biases |
| **Human evaluation** | Nuanced judgments | Expensive, doesn't scale |
| **Automated metrics** | Large-scale monitoring | May miss quality nuances |

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

class AgentEvaluator {
  constructor() {
    this.evaluationResults = [];
    this.latencyRecords = [];
    this.tokenUsage = { input: 0, output: 0 };
  }
  
  // Track response time
  async measureLatency(fn) {
    const start = Date.now();
    const result = await fn();
    const latency = Date.now() - start;
    this.latencyRecords.push(latency);
    return { result, latency };
  }
  
  // LLM-as-a-Judge: evaluate response quality
  async judgeResponse(question, response, criteria) {
    const judgeResponse = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are an expert evaluator. Score the response from 1-10 on these criteria:
            ${criteria.map((c, i) => `${i + 1}. ${c}`).join("\n")}
            
            Output ONLY valid JSON: {"score": 7, "scores": {"accuracy": 8, "helpfulness": 7}, "feedback": "..."}`
        },
        {
          role: "user",
          content: `Question: ${question}\n\nResponse to evaluate: ${response}`
        }
      ]
    });
    
    return JSON.parse(judgeResponse.choices[0].message.content);
  }
  
  // Evaluate an agent on a test set
  async evaluateAgent(agent, testCases) {
    console.log(`\nEvaluating agent on ${testCases.length} test cases...\n`);
    
    for (const testCase of testCases) {
      const { result: response, latency } = await this.measureLatency(
        () => agent(testCase.input)
      );
      
      const evaluation = await this.judgeResponse(
        testCase.input,
        response,
        ["accuracy", "helpfulness", "clarity"]
      );
      
      const result = {
        input: testCase.input,
        expectedOutput: testCase.expectedOutput,
        actualOutput: response,
        evaluation,
        latency,
        passed: evaluation.score >= 7
      };
      
      this.evaluationResults.push(result);
      
      console.log(`Test: "${testCase.input.substring(0, 50)}..."`);
      console.log(`  Score: ${evaluation.score}/10 | Latency: ${latency}ms | ${result.passed ? "✅ PASS" : "❌ FAIL"}`);
      console.log(`  Feedback: ${evaluation.feedback}`);
    }
  }
  
  // Generate evaluation report
  getReport() {
    const passRate = (this.evaluationResults.filter(r => r.passed).length / this.evaluationResults.length * 100).toFixed(1);
    const avgScore = (this.evaluationResults.reduce((sum, r) => sum + r.evaluation.score, 0) / this.evaluationResults.length).toFixed(1);
    const avgLatency = (this.latencyRecords.reduce((sum, l) => sum + l, 0) / this.latencyRecords.length).toFixed(0);
    
    return {
      totalTests: this.evaluationResults.length,
      passRate: `${passRate}%`,
      averageScore: `${avgScore}/10`,
      averageLatency: `${avgLatency}ms`,
      recommendation: parseFloat(passRate) >= 80 
        ? "Agent is performing well" 
        : "Agent needs improvement"
    };
  }
}

// Test
async function testEvaluation() {
  // Simple agent to evaluate
  const simpleAgent = async (question) => {
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: question }]
    });
    return response.choices[0].message.content;
  };
  
  const evaluator = new AgentEvaluator();
  
  await evaluator.evaluateAgent(simpleAgent, [
    { input: "What is the boiling point of water?", expectedOutput: "100°C or 212°F" },
    { input: "Explain photosynthesis simply", expectedOutput: "Plants convert light to energy" },
    { input: "What is 15% of 200?", expectedOutput: "30" }
  ]);
  
  console.log("\n📊 EVALUATION REPORT:");
  console.log(evaluator.getReport());
}

testEvaluation();
```

**Key Takeaways:**
- Always define clear evaluation metrics before deploying
- LLM-as-a-Judge enables scalable quality evaluation
- Track latency and token usage to manage costs
- Evaluate agent *trajectories* (the steps taken), not just final answers
- Re-evaluate regularly — performance drifts over time

---

### Chapter 20: Prioritization

**The simple idea:** Agents often face many tasks at once. Prioritization ensures they focus on what matters most — urgent, high-impact tasks first.

**Priority criteria:**
- **Urgency** — How time-sensitive is it?
- **Importance** — How much impact does it have?
- **Dependencies** — Does another task need this first?
- **Resources** — What's available right now?

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

class PrioritizationAgent {
  constructor() {
    this.taskQueue = [];
    this.completedTasks = [];
  }
  
  addTask(task) {
    this.taskQueue.push({ ...task, id: Date.now().toString(), status: "pending" });
  }
  
  // Use LLM to prioritize tasks
  async prioritizeTasks() {
    if (this.taskQueue.length === 0) return;
    
    const tasksJson = JSON.stringify(this.taskQueue.map(t => ({
      id: t.id,
      description: t.description,
      deadline: t.deadline,
      urgency: t.urgency,
      importance: t.importance
    })));
    
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a task prioritization expert. Given these tasks, return them sorted by priority (highest first).
            Consider: urgency (1-5), importance (1-5), deadline, and dependencies.
            Reply with a JSON array of task IDs in priority order: {"priority_order": ["id1", "id2", ...]}`
        },
        { role: "user", content: tasksJson }
      ]
    });
    
    const { priority_order } = JSON.parse(response.choices[0].message.content);
    
    // Reorder queue based on AI's prioritization
    this.taskQueue = priority_order
      .map(id => this.taskQueue.find(t => t.id === id))
      .filter(Boolean);
    
    console.log("\nPrioritized task order:");
    this.taskQueue.forEach((task, i) => {
      console.log(`  ${i + 1}. [${task.urgency}/${task.importance}] ${task.description}`);
    });
  }
  
  // Process tasks in priority order
  async processTasks() {
    await this.prioritizeTasks();
    
    while (this.taskQueue.length > 0) {
      const task = this.taskQueue.shift();
      console.log(`\nProcessing: "${task.description}"`);
      
      const response = await client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: "You are a task executor. Complete the given task efficiently." },
          { role: "user", content: `Complete this task: ${task.description}` }
        ]
      });
      
      task.result = response.choices[0].message.content.substring(0, 150);
      task.status = "completed";
      this.completedTasks.push(task);
      console.log(`  ✅ Done: ${task.result}...`);
    }
  }
}

async function testPrioritization() {
  const agent = new PrioritizationAgent();
  
  // Add tasks with different priority levels
  agent.addTask({ description: "Send weekly report to management", urgency: 5, importance: 4, deadline: "Today" });
  agent.addTask({ description: "Update project documentation", urgency: 2, importance: 3, deadline: "Next week" });
  agent.addTask({ description: "Fix critical production bug", urgency: 5, importance: 5, deadline: "ASAP" });
  agent.addTask({ description: "Review teammate's code PR", urgency: 3, importance: 3, deadline: "Tomorrow" });
  
  await agent.processTasks();
  
  console.log("\n📋 All tasks completed:", agent.completedTasks.length);
}

testPrioritization();
```

**Key Takeaways:**
- Prioritize before executing — don't just take tasks in arrival order
- Use urgency × importance matrix (like Eisenhower Matrix)
- Dynamic re-prioritization when new tasks arrive or situations change
- Agents can use LLMs to make nuanced priority decisions

---

### Chapter 21: Exploration and Discovery

**The simple idea:** Sometimes agents don't just solve known problems — they discover *new* knowledge, find novel solutions, and explore uncharted territory. This pattern is about building agents that can be genuinely curious and inventive.

**Real-world example — Google's AI Co-Scientist:**

Google built an AI agent that functions as a **research partner for scientists**. It:
- Reviews scientific literature
- Generates hypotheses
- Has agents debate and critique each other's ideas
- Ranks hypotheses using Elo scoring (like chess ratings)
- Keeps improving — it actually discovered new drug candidates for leukemia that were validated in lab experiments!

**JavaScript Example:**

```javascript
const { OpenAI } = require("openai");
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

class ExplorationAgent {
  constructor(domain) {
    this.domain = domain;
    this.discoveredIdeas = [];
    this.eloRatings = {}; // Track idea quality
  }
  
  // Generate initial hypotheses
  async generateHypotheses(topic, count = 5) {
    console.log(`\n💡 Generating ${count} hypotheses about: ${topic}`);
    
    const response = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: `You are a creative scientist in ${this.domain}. 
            Generate ${count} novel, testable hypotheses about the topic.
            Make them specific and actionable.
            Output as JSON: {"hypotheses": ["hypothesis 1", "hypothesis 2", ...]}`
        },
        { role: "user", content: topic }
      ]
    });
    
    const { hypotheses } = JSON.parse(response.choices[0].message.content);
    hypotheses.forEach((h, i) => {
      const id = `H${Date.now()}-${i}`;
      this.discoveredIdeas.push({ id, hypothesis: h, score: 1000 }); // Start at Elo 1000
      this.eloRatings[id] = 1000;
    });
    
    console.log("Generated hypotheses:");
    hypotheses.forEach((h, i) => console.log(`  ${i + 1}. ${h}`));
    return hypotheses;
  }
  
  // Debate two hypotheses and pick a winner
  async debateHypotheses(hypothesisA, hypothesisB) {
    const response = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: `You are a scientific debate judge. Evaluate two hypotheses and determine which is:
            - More testable
            - More novel
            - More likely to yield insights
            Output: {"winner": "A" or "B", "reason": "brief explanation"}`
        },
        {
          role: "user",
          content: `Hypothesis A: ${hypothesisA}\n\nHypothesis B: ${hypothesisB}`
        }
      ]
    });
    
    return JSON.parse(response.choices[0].message.content);
  }
  
  // Run a tournament to find the best hypotheses
  async runTournament() {
    console.log("\n🏆 Running hypothesis tournament...");
    const ideas = [...this.discoveredIdeas];
    
    // Simple round-robin tournament
    for (let i = 0; i < ideas.length; i++) {
      for (let j = i + 1; j < ideas.length; j++) {
        const result = await this.debateHypotheses(
          ideas[i].hypothesis,
          ideas[j].hypothesis
        );
        
        // Update Elo ratings
        const k = 32; // Elo K-factor
        const expectedA = 1 / (1 + Math.pow(10, (this.eloRatings[ideas[j].id] - this.eloRatings[ideas[i].id]) / 400));
        
        if (result.winner === "A") {
          this.eloRatings[ideas[i].id] += k * (1 - expectedA);
          this.eloRatings[ideas[j].id] -= k * expectedA;
        } else {
          this.eloRatings[ideas[i].id] -= k * expectedA;
          this.eloRatings[ideas[j].id] += k * (1 - expectedA);
        }
      }
    }
    
    // Rank by Elo
    const ranked = ideas
      .map(idea => ({ ...idea, finalScore: this.eloRatings[idea.id] }))
      .sort((a, b) => b.finalScore - a.finalScore);
    
    console.log("\n🥇 Top hypotheses by Elo rating:");
    ranked.forEach((idea, i) => {
      console.log(`  ${i + 1}. (${Math.round(idea.finalScore)}) ${idea.hypothesis}`);
    });
    
    return ranked;
  }
  
  // Evolve and improve top hypotheses
  async evolveTopIdeas(topN = 2) {
    const ranked = await this.runTournament();
    const topIdeas = ranked.slice(0, topN);
    
    console.log(`\n🧬 Evolving top ${topN} hypotheses...`);
    
    for (const idea of topIdeas) {
      const response = await client.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: "You are a scientific research synthesizer. Improve and refine this hypothesis to make it more specific, testable, and impactful."
          },
          {
            role: "user",
            content: `Original hypothesis: ${idea.hypothesis}\n\nProvide an improved version:`
          }
        ]
      });
      
      console.log(`\nOriginal: ${idea.hypothesis}`);
      console.log(`Evolved: ${response.choices[0].message.content}`);
    }
  }
}

async function testExploration() {
  const agent = new ExplorationAgent("materials science");
  
  await agent.generateHypotheses(
    "How can we improve the efficiency of solar panels using nanomaterials?",
    3 // Keep small for demo
  );
  
  await agent.evolveTopIdeas(1);
}

testExploration();
```

**Key Takeaways:**
- Exploration agents are proactive — they find new knowledge, not just answer questions
- Multi-agent debates improve idea quality (like peer review)
- Elo ranking from chess can be used to evaluate competing ideas
- Google's AI Co-Scientist used this approach to discover real drug candidates

---

## Quick Reference Cheat Sheet

| # | Pattern | One-Line Summary | Use When |
|---|---------|-----------------|----------|
| 1 | **Prompt Chaining** | Break big tasks into sequential steps | Task has multiple distinct stages |
| 2 | **Routing** | Direct requests to the right specialist | Different inputs need different handlers |
| 3 | **Parallelization** | Do independent tasks simultaneously | Multiple tasks don't depend on each other |
| 4 | **Reflection** | Review and improve your own output | Quality matters more than speed |
| 5 | **Tool Use** | Call external functions and APIs | Agent needs real-world data or actions |
| 6 | **Planning** | Make a plan before acting | Goal is complex with multiple steps |
| 7 | **Multi-Agent** | Team of specialists collaborate | Problem needs diverse expertise |
| 8 | **Memory** | Remember across conversations | Agent needs continuity over time |
| 9 | **Learning** | Improve from experience | Agent operates in dynamic environments |
| 10 | **MCP** | Universal standard for AI-to-tool connections | Need interoperability across tools/AI systems |
| 11 | **Goal Monitoring** | Track progress toward defined goals | Agent runs autonomously for extended periods |
| 12 | **Exception Handling** | Detect and recover from failures | Deploying in real-world environments |
| 13 | **Human-in-the-Loop** | Keep humans involved in critical decisions | High-stakes, ambiguous, or ethical decisions |
| 14 | **RAG** | Look up external knowledge before answering | Agent needs current or private information |
| 15 | **A2A** | Standard for agents talking to agents | Building cross-framework multi-agent systems |
| 16 | **Resource Optimization** | Use cheapest model that gets the job done | Cost or latency constraints |
| 17 | **Reasoning Techniques** | Think step by step (CoT, ReAct, ToT) | Complex problems requiring deep analysis |
| 18 | **Guardrails** | Safety systems to prevent harm | Any production deployment |
| 19 | **Evaluation** | Measure and improve agent performance | Before and during production deployment |
| 20 | **Prioritization** | Focus on the most important tasks first | Agent manages many competing tasks |
| 21 | **Exploration** | Discover new knowledge and solutions | Scientific research, creative tasks, innovation |

---

## Frameworks Used in the Book

| Framework | Best For | Key Feature |
|-----------|----------|-------------|
| **LangChain / LangGraph** | Chaining, routing, complex pipelines | LangChain Expression Language (LCEL) |
| **Crew AI** | Multi-agent collaboration | Role-based agent teams |
| **Google ADK** | Google ecosystem agents | Native Google tool integration |
| **OpenAI API** | Direct LLM access | Best tool calling support |

---

*This guide covers all 21 agentic design patterns from the book. Each pattern is a tool — the skill is knowing which tool to pick for which problem.*
