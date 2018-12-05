$(function () {
    'use strict';
    var hlp = $('#upload_info');
    var err = $('#upload_errors');
    var ldInd = $('#uploadIndicator');

    // Add files button logic
    $('.fileupload').fileupload({
        url: 'upload/',
        dataType: 'json',
        done: function (e, data) {
            err.empty();
            var errors = false;
            $.each(data.result.files, function (index, file) {
                if ( file.error ) {
                    $('</p>').text(file.error+' - '+file.name)
                             .appendTo(err);
                    errors = true;
                } else {
                    var type_field_name = file.type;

                    if (file.type == "cif"){
                        type_field_name = "pdb";
                    }

                    $('#textholder_id_'+type_field_name+'_file').text(file.name);
                    console.log(file.url);
                    $('#id_'+type_field_name+'_file').val(file.url);
                    if (type_field_name == 'mtz') {
                        $.each(file.labels, function (label_name, choices) {
                            var select = $('#id_mtz_label_'+label_name).empty();
                            var haschoices = false;
                            $.each(choices, function (index, choice) {
                                console.log(label_name + ' ' + choice);
                                select.append(
                                    $('<option></option>')
                                    .attr('value', choice)
                                    .text(choice)
                                    );
                                haschoices = true;
                            });
                            if (haschoices) { select.removeAttr('disabled'); }
                        });
                    }
                }
            });
            console.log(errors);
            if ( errors ) {
                //Not very responsive but smb will rather clear errors
                //than change the viewport size...
                //hlp.hide();
                err.show();
            } else {
                //hlp.show();
                err.empty();
                err.hide();
            }
        },
        progressall: function (e, data) {
            var progress = parseInt(data.loaded / data.total * 100, 10);
            $('#progress .progress-bar').css(
                'width',
                progress + '%'
            );
        },
        error: function (e, data) {
            var hlp = $('#upload_info');
            var err = $('#upload_errors');
            err.empty();
            $('</p>').text('Error. Unable to connect with the server. Try again in a moment')
                     .appendTo(err);
            hlp.hide();
            err.show();
        },
        start: function (e) {
            $('.uploadIndicator').css('visibility', 'visible');
        },
        stop: function (e) {
            $('.uploadIndicator').css('visibility', 'hidden');
        }
    }).prop('disabled', !$.support.fileInput)
        .parent().addClass($.support.fileInput ? undefined : 'disabled');


    // Fetch PDB button logic
    $('#fetchPDB').click(function() {
        var code = $('#id_pdb_code').val();
        if (code.length != 4) {
             $('</p>').text(code+': invalid PDB ID')
                             .appendTo(err);
            //hlp.hide();
            err.show();
        } else {
            $.getJSON(
                'pdb/'+code.toLowerCase(),
                '',
                function(data, response) {
                    if(data.error) {
                        err.empty();
                        $('</p>').text(code+': '+data.error)
                                 .appendTo(err);
                        //hlp.hide();
                        err.show();
                    } else {
                        err.hide();
                        //hlp.show();
                        var txt = 'Using data for PDB ID: '+ code.toUpperCase();
                        $('#textholder_id_pdb_file').text(txt);
                        $('#textholder_id_mtz_file').text(txt);
                    }
                }
            );
        }
    });

    //Application selection logic
    $('#div_id_application').change(function () {
        var inp = $(this).find('option:selected');
        console.log('clicked'+inp.val());
        $('#mtz_label_f_col').hide();
        $('#mtz_label_ph_col').hide();
        $('#mtz_label_fobs_col').hide();
        $('#mtz_label_sigf_col').hide();
        $('#mtz_label_free_col').hide();
        $('#mtz_label_diff_f_col').hide();
        $('#mtz_label_diff_ph_col').hide();
        switch (inp.val()) {
            case 'recognition':
                $('#mtz_label_f_col').hide();
                $('#mtz_label_ph_col').hide();
                $('#mtz_label_fobs_col').show();
                $('#mtz_label_sigf_col').show();

                $('#file_upload_block').show();
                $('#labels_block').show();
                $('#pdb_code_block').hide();

                $('#task-info-validation').hide();
                $('#task-info-recognition').show();
                break;
            case 'validation_pdb':
                $('#mtz_label_f_col').hide();
                $('#mtz_label_ph_col').hide();
                $('#mtz_label_fobs_col').show();
                $('#mtz_label_sigf_col').show();

                $('#file_upload_block').hide();
                $('#labels_block').hide();
                $('#pdb_code_block').show();

                $('#task-info-recognition').hide();
                $('#task-info-validation').show();
                break;
        }
    });
    $('#div_id_application').change();
});
