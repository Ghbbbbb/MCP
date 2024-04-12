<div align='center' ><font size='150'>Hospital Item Transport Dataset</font></div>

This is a dataset for machine-assisted hospital item transport, known as the Hospital Item Transport Dataset (HITD). It takes natural language instructions as input and generates low-level code to facilitate item transport tasks in a hospital setting. We provide both Chinese(`zh`) and English(`en`) versions of the HITD dataset for research purposes. It is important to note that all results discussed in this document are derived from the Chinese(`zh`) dataset. Additionally, within this project, we offer 1-shot English prompt (`prompt/en`) for researchers to validate results.

Each natural language instruction consists of three components: [*originating department*,*intermediate transport department*,*final destination department*]. The *originating department* and *intermediate transport department* are selected from a list of 10 common hospital departments, and the *final destination department* is chosen from "Logistics" or "General Services".

![Introduction of HITD](https://github.com/Ghbbbbb/MCP/main/assets/HITD.png)

## 1.Explanation of low-level code parameters:

**MOVP P="parameter1" OP="parameter2"**: This is a command for controlling the movement of the robot. *Parameter1* indicates the location to which movement is possible (type=init), and *parameter2* specifies the static operation to be executed (usually inputting the value 28000 to indicate a pickup operation). Example usage: *MOVP P = 1 OP = 28000* # Move to Cardiac Surgery and execute pickup operation.
**WHILE/ENDWHILE**: Executes commands following *WHILE* if conditions are met; otherwise, proceeds to commands following *ENDWHILE*.
**SET**: Assigns a value to a variable. Example: *SET #IMR.LP(10) 0* sets the parameter #IMR.LP(10) to 0.
**#GP(x)**: Represents a global variable indicating the quantity of items to be transported for a specific department (*x* representing one of the 10 departments). Initial values are all set to 0, and each item delivered increments this variable by 1. Departments are denoted by numbers 1 to 10 corresponding to: Cardiac Surgery, Internal Medicine, Obstetrics and Gynecology, Respiratory, Hematology, Surgery, Pediatrics, Thoracic Surgery, Gastrointestinal Surgery, and Gastroenterology.
**ADD X Y**: Performs addition operation X + Y and assigns the result back to X. Example: *ADD #GP(1) 1* increments the value of variable #GP(1) by 1.

## 2.The organization of the data directory is as follows:
```
.
├── en
│   ├── HITD.json
│   ├── HITD_MD.json
│   ├── HITD_MDP.json
│   ├── HITD_SD.json
│   └── robustness
│       ├── HITD_noise1.json
│       ├── HITD_noise2.json
│       ├── HITD_noise3.json
│       └── HITD_no_noise.json
├── README.md
└── zh
    ├── HITD.json
    ├── HITD_MD.json
    ├── HITD_MDP.json
    ├── HITD_SD.json
    └── robustness
        ├── HITD_noise1.json
        ├── HITD_noise2.json
        ├── HITD_noise3.json
        └── HITD_no_noise.json
```

### Explanation of each JSON file:

- **HITD.json**: Contains the complete HITD dataset with 1000 samples. Each sample is structured as follows:
```
{
    "content": str,
    "summary": str,
    "task": str,
    "code length": int
}
```
"content": User-inputted instruction.  
"summary": Corresponding low-level code.  
"task": Type of task, categorized as Single_department, Multi_department, or Multi_department_priority, indicating increasing task difficulty and code complexity.  
"code length": Length of the generated code.

- **HITD_MD.json**: Subset of HITD dataset containing samples with "task" as "Multi_department" (325 samples). Average code length is 399 characters, with 1-4 *originating departments* and 1-5 *intermediate transport departments*, without priority.

- **HITD_MDP.json**: Subset of HITD dataset containing samples with "task" as "Multi_department_priority" (331 samples). Average code length is 371 characters, with 1-4 *originating departments* and 1-5 *intermediate transport departments*, with priority.

- **HITD_SD.json**: Subset of HITD dataset containing samples with "task" as "Single_department" (344 samples). Average code length is 202 characters, with 1 *originating department* and 1-9 *intermediate transport departments*, without priority.

- **HITD_no_noise.json**: 200 randomly selected samples from HITD dataset, serving as test samples without noise interference.

- **HITD_noise1.json**: Perturbed version of HITD_no_noise.json with noise, which includes changes in expression while retaining the overall meaning (e.g., synonym substitution, rephrasing of sentence structure).

- **HITD_noise2.json**: Includes an additional transport department (e.g., Dermatology) compared to the original dataset.

- **HITD_noise3.json**: Randomly changes the sequence number of a department (e.g., changing the sequence number of Cardiac Surgery from 1 to 0).