def find_common_e(bcs, *bcs_lists):
    # Create set for bcs
    set_bcs = set(bcs)
        # Initialize the union set with the set of bcs
    union = set()
    
    # Iterate through the list of lists
    for bc in bcs_lists:
        # Convert current list to set
        set_bc = set(bc)
        
        # Update the union set with the current list
        union = union.union(set_bc)
    
    # Find the common elements with bcs
    common_e = set_bcs.intersection(union)
    common_e_list = list(common_e)
    
    return common_e_list