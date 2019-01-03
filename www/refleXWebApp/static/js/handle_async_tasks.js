
$(async function(){

                loader_id = document.getElementById('loader')
                loader_id.style.visibility = 'visible';

                final_result_id = document.getElementById('final_result')
                final_result_id.style.visibility = 'hidden';

                task_id = document.getElementById('task_id').innerHTML
                while_heartbit = true

                while(while_heartbit) {
                    $.ajax({
                        url: '/api/result/' + task_id,
                        success: function (data) {

                            if (data.status) {
                              loader_id.style.visibility = 'hidden';
                              final_result_id.style.visibility = 'visible';
                              while_heartbit = false;
                            }
                    }});
                    await sleep(2000);
                }
            });
