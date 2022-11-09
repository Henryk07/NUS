# NUS EE3731c

# Instruction of running the code

### For EE3731c CA2

for char2double.m and double2char.m

Function instruction:
double_array = char2double(char_array)

example:

```sh
char2double('Tough times never last but tough people do')
double2char([65 32 109 97 110 32 105 115 32 98 117 116 32 119])
```

for compute_transition_probability.m

Function instruction:
pr_trans = compute_transition_probability(input_txt)

example:

```sh
i.e. pr_trans(1,1)
```

for logn_pr_txt.m

Function instruction:
logn_pr = logn_pr_txt(frank_encrypted_txt, pr_trans)

example:

```sh
logn_pr_txt(frank_encrypted_txt, pr_trans)
```

for metropolis.m

Function instruction:
[accept_new_key, prob_accept] = metropolis(current_key, new_key, pr_trans, encrypted_txt)

example:

```sh
metropolis(frank_decrypt_key, mystery_decrypt_key, pr_trans, frank_encrypted_txt)
```

for mcmc_decrypt_text.m

Function instruction:
[decrypted_txt, decrypt_key] = mcmc_decrypt_text(encrypted_txt, pr_trans)

example:

```sh
mcmc_decrypt_text(frank_encrypted_txt, pr_trans)
```
