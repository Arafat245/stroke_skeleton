def val_test(participant_ids, k):
    
    step_val = 5
    step_test = 5
    block_size = step_val + step_test
    n_blocks = len(participant_ids) // block_size

    block_start = (k % n_blocks) * block_size
    # Last block gets remainder so all 155 participants are used
    is_last_block = (block_start == (n_blocks - 1) * block_size) and (len(participant_ids) % block_size != 0)
    block_end = len(participant_ids) if is_last_block else block_start + block_size

    if k < n_blocks:
        if is_last_block:
            validation_pids_list = participant_ids[block_start : block_start + step_val]
            test_pids_list = participant_ids[block_start + step_val : block_end]
        else:
            validation_pids_list = participant_ids[block_start : block_start + step_val]
            test_pids_list = participant_ids[block_start + step_val : block_start + block_size]
    else:
        if is_last_block:
            validation_pids_list = participant_ids[block_start + step_val : block_end]
            test_pids_list = participant_ids[block_start : block_start + step_val]
        else:
            validation_pids_list = participant_ids[block_start + step_val : block_start + block_size]
            test_pids_list = participant_ids[block_start : block_start + step_val]

    return validation_pids_list, test_pids_list        