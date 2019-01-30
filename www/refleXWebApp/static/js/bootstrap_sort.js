/*
    $(document).ready(function () {
        $('#final_table').DataTable({
            "order": [[ 3, "desc" ]],
            "bPaginate": false,
            "bLengthChange": false,
            "bFilter": false,
            "bInfo": false,
            "bAutoWidth": false,
            "aoColumns": [
                { "sType": "numeric" },
                null,
                null,
                { "sType": "numeric" }
            ]
    });
*/

$(document).ready(function () {
    $('#final_table').DataTable({
        "order": [[ 3, "desc" ]],
        "bPaginate": false,
        "bLengthChange": false,
        "bFilter": false,
        "bInfo": false,
        "bAutoWidth": false,
        "columnDefs": [
            { 
            "type": 'numeric', 
            "targets": -1 
            }
        ]
    });

$('.dataTables_length').addClass('bs-select');
});
