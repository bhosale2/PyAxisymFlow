def gen_periodic_boundary_ghost_comm(ghost_size):
    """Generates kernel that will communicate the ghosts at the periodic boundary."""
    assert ghost_size > 0 and isinstance(ghost_size, int), "invalid ghost size"

    def periodic_boundary_ghost_comm(field):
        """Communicate the field values at the ghosts on the periodic boundary."""
        # only communicating along periodic Z axis
        field[:, :ghost_size] = field[:, -2 * ghost_size : -ghost_size]
        field[:, -ghost_size:] = field[:, ghost_size : 2 * ghost_size]

    return periodic_boundary_ghost_comm
