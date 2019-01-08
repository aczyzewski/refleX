
$(async function(){

                loader_id = document.getElementById('loader')
                loader_id.style.visibility = 'visible';

                final_result_id = document.getElementById('final_result')
                final_result_id.style.visibility = 'hidden';

                task_id = document.getElementById('task_id').innerHTML
                while_heartbit = true

                prob_loop_scaterring_id = document.getElementById('prob_loop_scaterring').innerHTML
                prob_background_ring_id = document.getElementById('prob_background_ring').innerHTML
                prob_strong_background_id = document.getElementById('prob_strong_background').innerHTML

                }


                while(while_heartbit) {
                    $.ajax({
                        url: '/api/result/' + task_id,
                        success: function (data) {

                            if (data.status) {
                              loader_id.style.visibility = 'hidden';
                              final_result_id.style.visibility = 'visible';

                              prob_background_ring_id = data.background_ring;
                              prob_loop_scaterring_id = data.loop_scattering;
                              prob_strong_background_id = data.strong_background;

                              while_heartbit = false;


                    }});
                    await sleep(2000);
                }
            });
