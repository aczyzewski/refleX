   // <!-- show api on screen with click me button - just for tests -->

            $('.apireq').click( function() {
                $.ajax({
                    url : "http://localhost:8000/snippets/",
                    dataType: "json",
                    success : function (data) {
                        $('#test').text( data[0].status);
                        $('#loop_scattering').text( data[0].loop_scattering);
                        $('#background_ring').text( data[0].background_ring);
                        $('#strong_background').text( data[0].strong_background);
                        $('#diffuse_scattering').text( data[0].diffuse_scattering);
                        $('#artifact').text( data[0].artifact);
                        $('#ice_ring').text( data[0].ice_ring);
                        $('#non_uniform_detector').text( data[0].non_uniform_detector);

                    }
                });
            });
