# Tensor Toolbox Contribution Guide

## Getting Started

Before you can make a contribution, you may need to create a **fork** of this repository.
Then you will be able to make a merge request following the instructions below. Be sure to create a **branch** in your fork for your changes.
See the bottom of these instructionts for detailed instructions.

## Checklist

- [ ] **Fork** Create a fork of this repository and **create a new branch** for your changes.
- [ ] **Merge Request** Create a draft merge request from your new branch when you are ready for feedback.
  * [ ] **Grants Permission** Grant permissions for the Tensor Toolbox maintainers to make modifications to your branch.
  * [ ] **Description** Describe the changes your merge request will make. 
   - Is it adding new functionality? If so, what is it? How is it useful to the community?
   - Is it improving existing functionality? If so, please provide specifics of the amount of improvement and provide code in a comment on the merge request that demonstrates the improvements. Be sure to retain the old version of the code so that comparisons can be easily made. Please consider the broad set of use cases for a code when improving it. Is your improvement useful for every case or just some? Be sure that the improvements are only activated in appropriate contexts.
  * [ ] **Issues** link any relevant issues via `#N` where `N` is the issue number.
  * [ ] **Checklist** Provide _this_ checklist as the first comment on the merge request.
  
- [ ] **Help Comments** Create or update comments for the m-files, following the style of the existing files. Be sure to explain all code options.

- [ ] **HTML Documentation** For any major new functionality, please follow the following steps.
  - [ ] Add HTML documentation in the `doc\html` directory with the name `XXX_doc.html`.
  - [ ] Ensure random seeds are set explicitly so every run produces the same results.
  - [ ] Publish the documentation into `doc\html` via `cd doc; publish('XXX_doc.m','stylesheet','ttb.xsl')`.
  - [ ] Add a pointer to this documentation file in `doc\html\helptoc.xml`.
  - [ ] Add pointers in any related higher-level files, e.g., a new method for CP should be referenced in the `cp.html` file.
  - [ ] Add link to HTML documentation from help comments in function.
  
- [ ] **Tests** Create or update tests in the `tests` directory, especially for bug fixes and strongly encouraged for new code.

- [ ] **Contents** If new functions were added to a class, go to the `maintenance` directory and run `update_classlist('Class',XXX)` to add the new functions to the class XXX help information. If new functions were added at 
top level, go to `maintenance` and run `update_topcontents` to update the Contents.m file at the top level.

- [ ] **Release Notes** 
Update `README.md` (under "Changes from [MOST RECENT VERSION]") with any significant bug fixes or additions.

- [ ] **Contributors List**
Update `CONTRIBUTORS.md` with your name and a brief description of the contribution.

- [ ] **Pass All Tests**
Confirm that all tests (including existing tests) pass in `tests` directory.

## How to Fork & Create a New Branch

- Go to (Tensor Toolbox GITLAB site)[https://gitlab.com/tensors/tensor_toolbox].
- In the upper right, click the "Fork" button and follow the instructions (recommend naming the repo `tensor_toolbox_<username>_fork` and making it public).
- In your fork, go to Repository->Branches, select "New Branch" button in upper right
- In the "New Branch" screen, name your branch and choose "dev" in the "Create from" dropdown menu.

## How to Create a Merge Request

- Start in your fork.
- Go to Respoitory->Branches, select "Merge request" on the appropriate branch.
- On the "New Merge Request" screen, under "Target branch", select "Change branches"
- For the Target branch, enter `tensors/tensor_toolbox` and `dev`
- If your branch is public, under "Contribution", check the box that says "Allow commites from members who can merge to the target branch."

## How to Grant Permissions for your Merge Request

If your branch is *public*...
- Start on your private branch
- Go to Merge Requests and click on the appropriate request to open in
- Select "Edit"
- Scoll to **Contribuion** and select the **Allow commits from members who can merge to the target branch** checkbox. 

See [Collaborate on merge requests across forks](https://docs.gitlab.com/ee/user/project/merge_requests/allow_collaboration.html) for more information.

If your branch is *private*...

- Start on your private branch
- Go to Project Information->Members
- Choose the "Invite a group" option in the upper right
- Invite the group "tensors" and give the role "Developer"
