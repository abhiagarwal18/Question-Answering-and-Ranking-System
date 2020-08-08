$(function() {
    $('#s').click(function() {
       var query = $('#q').val();
       console.log(query);

      $.ajax({
            type: 'POST',
            url : "http://localhost:5000/search",
            data: {'question' : query},
            dataType: 'json',
            success: function(response){
                  // alert(response);
                  var parsed_data = JSON.stringify(response);
                  parse = JSON.parse(parsed_data);

                  var idx=5
                  if(parse["0"]=='BITS Pilani')
                     idx=1

                  var div = document.getElementsByClassName("box")[0];
                  var ans;
                  if(idx==1)
                  {
                     ans = {
                        0: parse["0"],
                        1: parse["1"],
                     }
                  }
                  else
                  {
                     ans = {
                        0: parse["0"],
                        1: parse["1"],
                        2: parse["2"],
                        3: parse["3"],
                        4: parse["4"],
                        5: parse["5"]
                     }
                  }
                  

                  let newNode = document.createElement('div');      
                  newNode.innerHTML = "<br />" + '<center><font size="4"><b>' + ans[0] + '</b></font></center>' + "<br />";
                  div.appendChild( newNode );
            
                  var i;
                  for(i = 0; i < idx; i++)
                  {
                     let newNode = document.createElement('div');      
                     newNode.innerHTML = "<br />" + '<a href="' + ans[i+1] + '" target = "_blank">' + ans[i+1] + '</a>' + "<br />";
                     
                     div.appendChild( newNode );
                     
                  }
            },
            error: function(jqXHR, textStatus, errorThrown){
               alert("The following error occured: "+ textStatus, errorThrown, jqXHR);
            }
           
         });
    });
  });
  
  document.addEventListener('DOMContentLoaded');
