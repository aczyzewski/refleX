function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
};

const setVal = (progress, val) => {
  progress.style.width = `${val}%`;
  progress.value = val;
};

const setBackground = (container, val) => {
  // container.style.background = `url(${val})`;
  container.style.background = val;
};

function move_progress_bar(iteration) {
  // grab progress and input
  const progress = document.getElementById('progress');
  const val = document.getElementById('val');
  const backgroundImgContainer = document.getElementById('loading-bg')

  // const beegees = ['static/img/blue.png','static/img/red.png','static/img/yellow.png','static/img/green.png','static/img/White.png']
  const beegees = ['#2196f3', '#84fd84', '#fadase', '#f44336', '#ffffff']
  const actionId = iteration;

    switch (parseInt(actionId)) {
      case 1:
        setVal(progress, 10);
        //setBackground(backgroundImgContainer, beegees[0])
        processing_text_id.innerHTML = "Transfering your photo";
        break;
      case 2:
        setVal(progress, 30);
        // setBackground(backgroundImgContainer, beegees[3])
        processing_text_id.innerHTML = "Anomalies searching ... ";
        break;
      case 3:
        setVal(progress, 50);
        // setBackground(backgroundImgContainer, beegees[1])
        processing_text_id.innerHTML = "Anomalies searching";
        break;
      case 4:
        setVal(progress, 70);
        // setBackground(backgroundImgContainer, beegees[2])
        processing_text_id.innerHTML = "Anomalies searching continues";
        break;
      case 5:
        setVal(progress, 90);
        // setBackground(backgroundImgContainer, beegees[3])
        processing_text_id.innerHTML = "Finishing your results!";
        break;
      case 6:
        //setVal(progress, 100);
        // setBackground(backgroundImgContainer, beegees[4])
        processing_text_id.innerHTML = "Here is your result table!";
        break;
    }
  };

// $("#table tr").each(function(){
// $(this).find("td:eq(3)").empty()
// });

var prob1;
var prob2;
var prob3;
var prob4;
var prob5;
var prob6;
var prob7;



function update_table() {
  var table = document.getElementById("final_table");
  for (var i = 0, row; row = table.rows[i]; i++) {
    //iterate through rows
    //rows would be accessed using the "row" variable assigned in the for loop
    var element = row.cells[3]
    if(element.tagName == 'TD') {

      str = element.innerHTML
      str = str.substring(0, str.length - 1);
      value = parseInt(str)

      if(value >= 50 && value < 75) {
        row.className = 'highlight_orange'
      }

      if (value >= 75) {
        row.className = 'highlight_green'
      }

    }
  }
}


$(async function(){
                task_id = document.getElementById('task_id').innerHTML
                processing_text_id = document.getElementById('processing_text')
                progress__container_id = document.getElementById('progress__container');

                json_result_id = document.getElementById('json_result');

                loader_id = document.getElementById('loading-bg')
                loader_id.style.visibility = 'visible';

                final_result_id = document.getElementById('final_result')
                final_result_id.style.display = 'none';

                var prob_loop_scaterring_id = document.getElementById('prob_loop_scaterring')

                prob_background_ring_id = document.getElementById('prob_background_ring')
                prob_strong_background_id = document.getElementById('prob_strong_background')
                prob_diffuse_scattering_id = document.getElementById('prob_diffuse_scattering')
                prob_artifact_id = document.getElementById('prob_artifact')
                prob_ice_ring_id = document.getElementById('prob_ice_ring')
                prob_non_uniform_detector_id = document.getElementById('prob_non_uniform_detector')

                actionType = parseInt('1')

                while_heartbit = true
                while(while_heartbit) {

                    $.ajax({
                        url: '/api/result/' + task_id,
                        success: function (data) {

                            if (data.status) {

                              loader_id.style.display = 'none';

                              move_progress_bar(6);
                              sleep(500);
                              //progress__container_id.style.display = 'none';
                              final_result_id.style.display = 'block';

                              prob_loop_scaterring_id.innerHTML =  (parseFloat(data.loop_scattering) * 100).toString() + "%";
                              prob_background_ring_id.innerHTML = (parseFloat(data.background_ring) * 100).toString() + "%";
                              prob_diffuse_scattering_id.innerHTML = (parseFloat(data.diffuse_scattering) * 100).toString() + "%";
                              prob_strong_background_id.innerHTML = (parseFloat(data.strong_background) * 100).toString() + "%";
                              prob_ice_ring_id.innerHTML = (parseFloat(data.ice_ring) * 100).toString() + "%";
                              prob_artifact_id.innerHTML = (parseFloat(data.artifact) * 100).toString() + "%";
                              prob_non_uniform_detector_id.innerHTML =  (parseFloat(data.non_uniform_detector) * 100).toString() + "%";

                              prob1 = data.loop_scattering;
                              prob2 = data.background_ring;

                              update_table()

                              console.log("END!")
                              while_heartbit = false;
                            }}

                })

              await sleep(3400);
              //move_progress_bar(actionType)
              //actionType = actionType + 1;
            }

          });
