
            //<!-- handling of task 1 status -->
            $(async function(){
                status_element = document.getElementById('kolko_status')
                task_id = document.getElementById('task_id').innerHTML
                loader_id = document.getElementById('loader')
                final_result_id = document.getElementById('final_result')
                final_result_id.style.visibility = 'hidden';
                while_heartbit = true
                while(while_heartbit) {
                    $.ajax({
                        url: '/result/' + task_id,
                        success: function (data) {
                            if (data.includes("SUCCESS")) {
                                status_element.innerHTML = 'NIE';
                                loader_id.style.visibility = 'hidden';
                                final_result_id.style.visibility = 'visible';
                                while_heartbit = false;
                            }
                            else
                                status_element.innerHTML = 'TAK';
                        }
                    });
                    await sleep(2000);
                }
            });