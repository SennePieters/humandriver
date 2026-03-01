# Gemini Code Assist Instructions: HumanDriver Project

## 🎯 Project Context
**HumanDriver** is a Python application that uses CDP (Chrome DevTools Protocol) commands to control mouse and keyboard interactions. 
* **Primary Goal:** All automated actions must appear indistinguishable from a real human user. 
* **Key Concepts:** Evasion of bot detection, physics-based mouse movements (WindMouse algorithm, gravity, jitter), bursty typing rhythms with realistic error correction/backspacing, and direct CDP integration for asynchronous input dispatch.

## 📐 Global Coding Rules
1. **Diagram-Driven Development:** This project strictly uses Mermaid.js diagrams as the source of truth for all logic and architecture. 
2. **No Code Before Logic:** Never write or modify Python code for a new feature until the Mermaid diagram representing that logic has been updated and approved by the user.
3. **CDP over High-Level Wrappers:** Prefer raw CDP commands and asynchronous execution (`asyncio`) over blocking, high-level automation wrappers unless specified otherwise.
4. **The "Human" Constraint:** When writing logic, always account for human imperfections (e.g., adding randomized micro-delays, jitter, or imperfect target clicks). 

## 🔄 The Daily Workflow (Strict Adherence Required)
When I ask to add a feature or change logic, you must follow these steps in order:

* **Step 1:** Read the Mermaid diagram in the `docs/` folder.
* **Step 2:** Update the Mermaid text to reflect the new logic. Ensure the flow accounts for "humanlike" delays or CDP event listeners.
* **Step 3:** Present *only* the updated Mermaid code so I can perform a visual sanity check in draw.io. **Do not write the Python code yet.**
* **Step 4:** Wait for my explicit approval.
* **Step 5:** Once I say the diagram is approved, write the Python code to perfectly match the new Mermaid logic. Ensure the code is asynchronous, highly optimized, and mathematically sound for human simulation.
