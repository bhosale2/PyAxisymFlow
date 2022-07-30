import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from set_sim_params import grid_size_z, grid_size_r


def vtk_init():
    """
    initialises the writer
    """
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(grid_size_z, grid_size_r, 1)
    if vtk.VTK_MAJOR_VERSION <= 5:
        vtk_image_data.SetNumberOfScalarComponents(1)
        vtk_image_data.SetScalarTypeToDouble()
    else:
        vtk_image_data.AllocateScalars(vtk.VTK_DOUBLE, 1)

    temp_vtk_array = vtk.vtkDoubleArray()

    writer = vtk.vtkXMLImageDataWriter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInputConnection(vtk_image_data.GetProducerPort())
    else:
        writer.SetInputData(vtk_image_data)

    return vtk_image_data, temp_vtk_array, writer


def vtk_write(
    filename, vtk_image_data, temp_vtk_array, writer, field_names_list, field_list
):
    """
    vtk writer: see examples in case/ folder
    """
    temp_vtk_array.SetName("softy")
    field_count = len(field_names_list)
    temp_vtk_array.SetNumberOfComponents(field_count)
    temp_vtk_array.SetNumberOfTuples(grid_size_z * grid_size_r)
    for i in range(0, field_count):
        temp_vtk_array.SetComponentName(i, field_names_list[i])
        temp_vtk_array.CopyComponent(
            i,
            dsa.numpyTovtkDataArray(
                field_list[i].reshape(
                    -1,
                )
            ),
            0,
        )

    for i_array in range(vtk_image_data.GetPointData().GetNumberOfArrays()):
        vtk_image_data.GetPointData().RemoveArray(i_array)
    vtk_image_data.GetPointData().AddArray(temp_vtk_array)

    writer.SetFileName(filename)
    writer.Write()
