$(document).ready(function () {
$('#final_table').DataTable({
"order": [[ 3, "desc" ]],
"bPaginate": false,
"bLengthChange": false,
"bFilter": false,
"bInfo": false,
"bAutoWidth": false
});


$('.dataTables_length').addClass('bs-select');
});
